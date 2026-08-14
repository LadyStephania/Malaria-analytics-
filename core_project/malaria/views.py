import csv
import io
import json
import logging
import math
import datetime
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from concurrent.futures import ThreadPoolExecutor
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.cache import cache
from django.db import IntegrityError
from django.db.models import Sum, Avg, Max, Min, Count
from django.http import HttpResponse
from .models import SystemUser, IntegratedMalariaData, ZambianDistrict

logger = logging.getLogger(__name__)

# ================= STATISTICAL HELPERS =================
# Pure stdlib implementations (no numpy/scipy dependency required).

def _pearson_r(xs, ys):
    """Pearson correlation coefficient between two equal-length numeric sequences."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    return cov / denom if denom else None


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _p_value_from_r(r, n):
    """Two-tailed p-value for a Pearson r via the Fisher z-transformation."""
    if r is None or n < 4:
        return None
    if abs(r) >= 1:
        return 0.0
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return 2 * (1 - _norm_cdf(abs(z)))


def _classify_risk(rainfall_mm, temp_c):
    """Shared rule-of-thumb vector-breeding risk classifier (rainfall + temperature window)."""
    if rainfall_mm > 10.0 and 24.0 <= temp_c <= 30.0:
        return 'Critical Outbreak Risk 🔥', 'danger'
    elif rainfall_mm > 2.0 or 20.0 <= temp_c <= 32.0:
        return 'Moderate Risk Alert ⚠️', 'warning'
    return 'Low Stable Risk ✅', 'success'


_RISK_RANK = {'success': 0, 'warning': 1, 'danger': 2}
_BADGE_COLOR = {'danger': '#dc3545', 'warning': '#f59e0b', 'success': '#16a34a'}


def _classify_burden(cases, population):
    """
    Population-adjusted case-burden classifier — cases per 10,000 residents,
    cumulative over the uploaded reporting period, when population is known;
    falls back to raw case-count thresholds otherwise. Shared by the Dashboard
    hotspot map and Decision Support so the two views always agree.
    Returns (label, badge, incidence_per_10k_or_None).
    """
    population = population or 0
    if population > 0:
        incidence = round((cases / population) * 10000, 1)
        if incidence >= 50:
            return 'High Burden', 'danger', incidence
        elif incidence >= 15:
            return 'Moderate Burden', 'warning', incidence
        return 'Low Burden', 'success', incidence
    if cases >= 200:
        return 'High Burden', 'danger', None
    elif cases >= 50:
        return 'Moderate Burden', 'warning', None
    return 'Low Burden', 'success', None


def _province_populations():
    """
    {province: total_population}, summed from each province's real district
    populations (ZambianDistrict.population, sourced from the 2022 census —
    see the district-coordinate fix earlier). There's no separate
    province-population field; a province's population IS the sum of its
    districts', so this is computed directly from ZambianDistrict rather than
    through IntegratedMalariaData (which would double-count a district's
    population once per reporting period it has a record for).
    """
    rollup = (
        ZambianDistrict.objects
        .values('province')
        .annotate(population=Sum('population'))
    )
    return {row['province']: row['population'] or 0 for row in rollup}


# ---- Getis-Ord Gi* hotspot detection ----
# We only have district centroids (latitude/longitude), not polygon boundaries —
# there's no shapefile in this project — so "neighbor" can't mean shares-a-border
# the way GIS hot-spot tools usually define it. K-nearest-neighbor weights (each
# district's k closest other districts by straight-line distance) are the
# standard substitute when only point locations are available, and are one of
# the built-in weighting options in real GIS hot-spot tools (e.g. ArcGIS's own
# Hot Spot Analysis) precisely for this situation.
_GI_STAR_BANDS = [
    (1.65, None),
    (1.96, '90% confidence'),
    (2.58, '95% confidence'),
    (float('inf'), '99% confidence'),
]


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    r = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _confidence_label(abs_z):
    for ceiling, label in _GI_STAR_BANDS:
        if abs_z < ceiling:
            return label
    return '99% confidence'


def _getis_ord_gi_star(points, k=6):
    """
    points: list of {'id', 'name', 'lat', 'lon', 'value'} — one per district,
    'value' being whatever's being tested for clustering (incidence per 10,000
    here). Returns the same list with 'z_score', 'p_value', 'tier'
    ('hot'/'cold'/'not_significant'), and 'confidence' (None or a confidence
    label) added per point.

    Method: binary k-nearest-neighbor spatial weights (w_ij = 1 if j is one of
    i's k nearest neighbors by great-circle distance, or j == i; 0 otherwise —
    Gi*, unlike plain Gi, includes the location itself in its own neighborhood).
    For each location i:

        Gi* = [ sum(x_j for j in N(i)) - x_bar * W_i ]
              / ( S * sqrt( (n*W_i - W_i^2) / (n-1) ) )

    where W_i = |N(i)| = k+1 for every i (binary weights, so sum(w_ij) ==
    sum(w_ij^2) == W_i), x_bar/S are the global mean/population-std-dev of the
    analysis variable, and n is the number of locations. This is the standard
    Getis-Ord Gi* statistic — a z-score: values further from 0 than +-1.65/
    +-1.96/+-2.58 indicate the location's own value AND its neighbors' values
    are consistently high (hot spot) or low (cold spot) at 90/95/99%
    confidence — a real statistical claim about spatial clustering, not just
    "this one point looks high."
    """
    n = len(points)
    if n < k + 2:
        return None  # not enough locations for k neighbors + a meaningful comparison set

    values = [p['value'] for p in points]
    x_bar = sum(values) / n
    variance = sum((v - x_bar) ** 2 for v in values) / n  # population variance
    s = math.sqrt(variance)
    if s == 0:
        return None  # every location has the identical value — no variation to cluster

    # k nearest neighbors per point, by great-circle distance.
    neighbor_sets = []
    for i, p in enumerate(points):
        dists = sorted(
            ((j, _haversine_km(p['lat'], p['lon'], q['lat'], q['lon'])) for j, q in enumerate(points) if j != i),
            key=lambda pair: pair[1]
        )
        neighbor_idx = {j for j, _dist in dists[:k]}
        neighbor_idx.add(i)  # Gi* includes the location itself
        neighbor_sets.append(neighbor_idx)

    results = []
    for i, p in enumerate(points):
        w_i = len(neighbor_sets[i])
        local_sum = sum(values[j] for j in neighbor_sets[i])
        numerator = local_sum - x_bar * w_i
        denominator = s * math.sqrt((n * w_i - w_i ** 2) / (n - 1))
        z = numerator / denominator if denominator else 0.0
        p_value = 2 * (1 - _norm_cdf(abs(z)))

        abs_z = abs(z)
        if abs_z < 1.65:
            tier, confidence = 'not_significant', None
        elif z > 0:
            tier, confidence = 'hot', _confidence_label(abs_z)
        else:
            tier, confidence = 'cold', _confidence_label(abs_z)

        results.append({**p, 'z_score': round(z, 2), 'p_value': round(p_value, 4), 'tier': tier, 'confidence': confidence})

    return results


def _fetch_forecast_days(lat, lon):
    """
    Returns a list of up to 14 {date, rain, temp, badge} dicts — the raw daily
    forecast for the given coordinates, each day classified via _classify_risk —
    or None if the external API is unreachable. Cached 6 hours per rounded
    location so the Dashboard/Weather/Decision pages sharing a location don't
    each re-hit the API on every view.
    """
    cache_key = f"forecast_raw_{round(lat, 2)}_{round(lon, 2)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    api_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ["temperature_2m_max", "rain_sum"],
        "timezone": "Africa/Lusaka",
        "forecast_days": 14,
    }
    try:
        response = requests.get(api_url, params=params, timeout=10)
        api_data = response.json().get('daily', {})
        dates = api_data.get('time', [])
        max_temps = api_data.get('temperature_2m_max', [])
        rain_sums = api_data.get('rain_sum', [])

        days = []
        for i in range(len(dates)):
            rain_val = rain_sums[i] if rain_sums[i] is not None else 0.0
            temp_val = max_temps[i] if max_temps[i] is not None else 26.0
            _label, badge = _classify_risk(rain_val, temp_val)
            days.append({'date': dates[i], 'rain': rain_val, 'temp': temp_val, 'badge': badge})

        cache.set(cache_key, days, timeout=6 * 60 * 60)
        return days
    except Exception:
        return None


def _forecast_risk_summary(days):
    """Worst-case badge over the next 7 days (the actionable near-term window), plus supporting day counts."""
    if not days:
        return None
    near_term = days[:7]
    worst_rank = max(_RISK_RANK[d['badge']] for d in near_term)
    worst_badge = next(b for b, rank in _RISK_RANK.items() if rank == worst_rank)
    return {
        'badge': worst_badge,
        'danger_days_7': sum(1 for d in near_term if d['badge'] == 'danger'),
        'warning_days_7': sum(1 for d in near_term if d['badge'] == 'warning'),
        'danger_days_14': sum(1 for d in days if d['badge'] == 'danger'),
    }


_TIER_META = {
    'critical': {'risk_badge': 'danger', 'icon': '🚨', 'risk_level': 'Critical — Escalating Forecast & Active Outbreak'},
    'active': {'risk_badge': 'danger', 'icon': '🩺', 'risk_level': 'Active Outbreak Response Required'},
    'preemptive': {'risk_badge': 'warning', 'icon': '🌧️', 'risk_level': 'Pre-Emptive Deployment Advised'},
    'watch': {'risk_badge': 'warning', 'icon': '👁️', 'risk_level': 'Elevated Watch'},
    'stable': {'risk_badge': 'success', 'icon': '✅', 'risk_level': 'Stable — Routine Monitoring'},
}


def _combine_decision(forecast_rank, burden_rank):
    """
    Two-factor decision rule: escalate on EITHER a worsening forecast (breeding
    conditions ripening over the next 7 days) OR high confirmed-case burden (an
    outbreak already under way) — and distinguishes PRE-EMPTIVE prevention from
    ACTIVE outbreak response, so the recommended action matches whichever factor
    is actually driving the risk rather than blending them into one generic tier.
    Ranks are 0=Low, 1=Moderate, 2=High; forecast_rank is None if the external
    forecast API was unreachable, in which case the decision falls back to
    case burden alone.
    """
    if forecast_rank is None:
        return 'active' if burden_rank == 2 else ('watch' if burden_rank == 1 else 'stable')
    if forecast_rank == 2 and burden_rank == 2:
        return 'critical'
    if forecast_rank == 2:
        return 'preemptive'
    if burden_rank == 2:
        return 'active'
    if forecast_rank == 1 or burden_rank == 1:
        return 'watch'
    return 'stable'


# ================= 1. GATEWAY VIEW: SECURITY LOGIN =================
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    if request.method == 'POST':
        u = request.POST.get('username')
        p = request.POST.get('password')
        user = authenticate(request, username=u, password=p)
        if user is not None:
            login(request, user)
            messages.success(request, f"Access Granted. Logged in as {user.get_role_display()}.")
            return redirect('dashboard')
        else:
            messages.error(request, "Invalid credentials. Access Denied.")
    return render(request, 'login.html')

# ================= 2. SESSION LOGOUT UTILITY =================
@login_required
def logout_view(request):
    logout(request)
    return redirect('login')

# ================= 3. HOME VIEW: OVERVIEW DASHBOARD =================
@login_required
def dashboard_view(request):
    totals = IntegratedMalariaData.objects.aggregate(
        total_confirmed=Sum('rdt_confirmations'),
        latest_date=Max('date'),
        earliest_date=Min('date'),
    )
    total_confirmed = totals['total_confirmed'] or 0
    has_data = totals['latest_date'] is not None
    districts_reporting = IntegratedMalariaData.objects.values('district_id').distinct().count()
    total_districts = ZambianDistrict.objects.count()

    # A single label for how much reporting history the cumulative stats below
    # actually span, so a number like the hotspot totals or critical-node count
    # doesn't read as "this year's cases" when it's really a multi-year sum.
    if totals['earliest_date'] and totals['latest_date']:
        earliest_year, latest_year = totals['earliest_date'].year, totals['latest_date'].year
        data_year_range = str(latest_year) if earliest_year == latest_year else f"{earliest_year}–{latest_year}"
    else:
        data_year_range = None

    # Roll up every district that has records, for both the "critical node"
    # headline stat and the Leaflet hotspot map.
    district_rollup = (
        IntegratedMalariaData.objects
        .values('district_id', 'district__name', 'district__latitude', 'district__longitude', 'district__population')
        .annotate(
            cases=Sum('rdt_confirmations'),
            avg_rainfall=Avg('rainfall_mm'),
            avg_temp=Avg('avg_temperature_c'),
        )
    )

    # Hotspot tiers are population-adjusted (cases per 10,000 residents, cumulative
    # over the uploaded reporting period) whenever a district's population is known —
    # this is the standard epidemiological approach, since raw case counts alone
    # over-flag large/urban districts and under-flag small/rural ones. Districts with
    # no population on file fall back to raw case-count thresholds instead.
    priority_points, fallback_points = [], []
    for row in district_rollup:
        cases = row['cases'] or 0
        population = row['district__population'] or 0

        risk_label, badge, incidence = _classify_burden(cases, population)
        marker_color = _BADGE_COLOR[badge]

        point = {
            'id': row['district_id'],
            'name': row['district__name'],
            'lat': row['district__latitude'],
            'lon': row['district__longitude'],
            'cases': cases,
            'population': population,
            'incidence_per_10k': incidence,
            'risk': risk_label,
            'color': marker_color,
        }
        (priority_points if population > 0 else fallback_points).append(point)

    # Population-known districts are ranked by incidence rate (the epidemiologically
    # meaningful order); districts with unknown population are ranked by raw count
    # and listed after, since their tier is only a rough proxy.
    priority_points.sort(key=lambda p: p['incidence_per_10k'], reverse=True)
    fallback_points.sort(key=lambda p: p['cases'], reverse=True)
    map_points = priority_points + fallback_points

    alert_district = map_points[0]['name'] if map_points else 'No Data Yet'

    # How many reporting districts fall in each burden tier — reuses the exact
    # same classification as the map/ranking above, just counted instead of
    # listed, for a single-glance "how many places are in trouble" summary.
    tier_order = ['danger', 'warning', 'success']
    tier_labels = {'danger': 'High', 'warning': 'Moderate', 'success': 'Low'}
    tier_counts = {badge: 0 for badge in tier_order}
    for point in map_points:
        badge = next(b for b, color in _BADGE_COLOR.items() if color == point['color'])
        tier_counts[badge] += 1
    burden_tier_summary = [
        {'label': tier_labels[badge], 'count': tier_counts[badge], 'color': _BADGE_COLOR[badge]}
        for badge in tier_order
    ]

    # Confirmed cases grouped by province — a coarser, real aggregate for anyone
    # who wants a provincial view rather than scanning all 116 districts.
    # Population-adjusted the same way the district hotspot map is: cases per
    # 10,000 residents, using each province's real summed district population.
    province_populations = _province_populations()
    province_rollup = list(
        IntegratedMalariaData.objects
        .values('district__province')
        .annotate(cases=Sum('rdt_confirmations'), districts=Count('district_id', distinct=True))
        .order_by('-cases')
    )
    for row in province_rollup:
        pop = province_populations.get(row['district__province'], 0)
        row['population'] = pop
        row['incidence_per_10k'] = round(row['cases'] / pop * 10000, 1) if pop else None

    # Top 8 districts by the same rank the hotspot list uses, for a compact bar
    # chart alongside it.
    top_districts = map_points[:8]

    # Last 8 reported epidemiological weeks, oldest -> newest, for the trend strip.
    trend_rows = list(
        IntegratedMalariaData.objects
        .values('epi_week', 'reporting_year')
        .annotate(cases=Sum('rdt_confirmations'))
        .order_by('-reporting_year', '-epi_week')[:8]
    )
    trend_rows.reverse()

    # Some imports (e.g. annual NMEC/DHIS2 exports) carry no real intra-year
    # granularity — every record lands on the same placeholder epi_week. Labeling
    # those points "W1, W1, W1..." would imply a weekly resolution the data doesn't
    # have, so detect that case and label by year instead.
    is_annual_cadence = len({r['epi_week'] for r in trend_rows}) <= 1
    if is_annual_cadence:
        trend_labels = [str(r['reporting_year']) for r in trend_rows]
        trend_title = 'Year-over-Year Confirmed Case Trend'
    else:
        trend_labels = [f"W{r['epi_week']}" for r in trend_rows]
        trend_title = 'Recent Confirmed Case Trend (by Epi Week)'

    context = {
        'confirmed_rdt': total_confirmed,
        'districts_reporting': districts_reporting,
        'total_districts': total_districts,
        'alert_district': alert_district,
        'active_user_role': request.user.get_role_display(),
        'has_data': has_data,
        'latest_date': totals['latest_date'],
        'data_year_range': data_year_range,
        'map_points': map_points,
        'map_points_json': json.dumps(map_points),
        'data_year_range_json': json.dumps(data_year_range),
        'trend_labels_json': json.dumps(trend_labels),
        'trend_cases_json': json.dumps([r['cases'] or 0 for r in trend_rows]),
        'trend_title': trend_title,
        'trend_is_annual': is_annual_cadence,
        'burden_tier_summary': burden_tier_summary,
        'burden_tier_labels_json': json.dumps([t['label'] for t in burden_tier_summary]),
        'burden_tier_counts_json': json.dumps([t['count'] for t in burden_tier_summary]),
        'burden_tier_colors_json': json.dumps([t['color'] for t in burden_tier_summary]),
        'province_rollup': province_rollup,
        'top_districts': top_districts,
        'top_districts_labels_json': json.dumps([d['name'] for d in top_districts]),
        'top_districts_cases_json': json.dumps([d['cases'] for d in top_districts]),
        'top_districts_ids_json': json.dumps([d['id'] for d in top_districts]),
    }
    return render(request, 'dashboard.html', context)

# ================= 3b. YEAR DRILL-DOWN: EVERY DISTRICT, ONE YEAR =================
# The trend chart's x-axis is a year (a national aggregate) — clicking a point on
# it can only identify "which year", not "which district", since the chart has
# no district dimension. This view is the natural next step: given a year,
# break the national total back down by district so a specific district's count
# for that year can actually be read off.
@login_required
def year_breakdown_view(request, year):
    rows = list(
        IntegratedMalariaData.objects
        .filter(reporting_year=year)
        .values('district_id', 'district__name')
        .annotate(cases=Sum('rdt_confirmations'))
        .order_by('-cases')
    )

    # Same year's cases, one level up — by province, population-adjusted, so
    # "how many people got malaria in a province in this specific year, against
    # how many people live there" is answerable without averaging across years.
    province_populations = _province_populations()
    province_rows = list(
        IntegratedMalariaData.objects
        .filter(reporting_year=year)
        .values('district__province')
        .annotate(cases=Sum('rdt_confirmations'), districts=Count('district_id', distinct=True))
        .order_by('-cases')
    )
    for row in province_rows:
        pop = province_populations.get(row['district__province'], 0)
        row['population'] = pop
        row['incidence_per_10k'] = round(row['cases'] / pop * 10000, 1) if pop else None

    context = {
        'year': year,
        'rows': rows,
        'total_cases': sum(r['cases'] or 0 for r in rows),
        'district_labels_json': json.dumps([r['district__name'] for r in rows]),
        'district_cases_json': json.dumps([r['cases'] or 0 for r in rows]),
        'district_ids_json': json.dumps([r['district_id'] for r in rows]),
        'province_rows': province_rows,
    }
    return render(request, 'year_breakdown.html', context)

# ================= 3c. STATISTICAL HOTSPOT DETECTION (GETIS-ORD GI*) =================
_GI_K_OPTIONS = range(4, 11)
_GI_DEFAULT_K = 6


@login_required
def hotspot_view(request):
    try:
        k = int(request.GET.get('k', _GI_DEFAULT_K))
    except (TypeError, ValueError):
        k = _GI_DEFAULT_K
    k = max(4, min(10, k))

    # Same cumulative population-adjusted incidence the Dashboard hotspot map
    # and Decision Support use, so this view's "hot spot" and the simple
    # threshold map's "high burden" are directly comparable, not two
    # unrelated numbers.
    district_rollup = (
        IntegratedMalariaData.objects
        .values('district_id', 'district__name', 'district__latitude', 'district__longitude', 'district__population')
        .annotate(cases=Sum('rdt_confirmations'))
    )

    points = []
    for row in district_rollup:
        population = row['district__population'] or 0
        if population <= 0 or row['district__latitude'] is None or row['district__longitude'] is None:
            continue  # Gi* needs a real value and a real location for every point — can't include a district missing either
        cases = row['cases'] or 0
        incidence = round(cases / population * 10000, 1)
        points.append({
            'id': row['district_id'],
            'name': row['district__name'],
            'lat': row['district__latitude'],
            'lon': row['district__longitude'],
            'value': incidence,
            'cases': cases,
            'population': population,
        })

    results = _getis_ord_gi_star(points, k=k) if points else None
    excluded_count = district_rollup.count() - len(points)

    if results:
        results.sort(key=lambda r: -abs(r['z_score']))
        hot_spots = [r for r in results if r['tier'] == 'hot']
        cold_spots = [r for r in results if r['tier'] == 'cold']
        tier_color = {'hot': '#dc3545', 'cold': '#0d6efd', 'not_significant': '#94a3b8'}
        for r in results:
            r['color'] = tier_color[r['tier']]
    else:
        hot_spots, cold_spots = [], []

    context = {
        'k': k,
        'k_options': _GI_K_OPTIONS,
        'results': results,
        'results_json': json.dumps(results) if results else '[]',
        'hot_spots': hot_spots,
        'cold_spots': cold_spots,
        'n_analyzed': len(points),
        'n_excluded': excluded_count,
        'n_not_significant': len(points) - len(hot_spots) - len(cold_spots),
        'has_data': bool(results),
    }
    return render(request, 'hotspot_analysis.html', context)

# ================= 4. ANALYTICS CORRELATION ENGINE VIEW =================
_DRIVER_FIELDS = {
    'rainfall': ('rainfall_mm', 'Rainfall (mm)', 'rainfall', '#0d6efd'),
    'temperature': ('avg_temperature_c', 'Avg. Temperature (°C)', 'temperature', '#f59e0b'),
}
# Model-field defaults (models.py) stamped onto any row whose import didn't supply
# real weather — e.g. the NMEC/DHIS2 case-only CSV. Used below to detect when the
# correlation sample is dominated by placeholder values rather than real readings.
_DRIVER_DEFAULTS = {'rainfall_mm': 0.0, 'avg_temperature_c': 25.0}

# (upper bound, plain-language strength word) — standard rule-of-thumb bands for
# interpreting |r|, used to translate the raw coefficient into a phrase a reader
# doesn't need a statistics background to understand.
_CORRELATION_BANDS = [
    (0.1, 'no real'),
    (0.3, 'a weak'),
    (0.5, 'a moderate'),
    (0.7, 'a strong'),
    (1.01, 'a very strong'),
]


def _describe_correlation(r_value, p_value, plain_driver, lag_periods, cadence_unit, sample_size):
    """
    Translates the raw Pearson r / p-value into a one-sentence, jargon-free
    summary plus short plain-language captions for each stat card — so a reader
    with no statistics background gets the meaning without the numbers explained
    to them. The numbers themselves stay on the page (smaller, secondary) for
    anyone who wants them.
    """
    if r_value is None:
        return None

    abs_r = abs(r_value)
    strength_word = next(word for ceiling, word in _CORRELATION_BANDS if abs_r < ceiling)
    lag_phrase = f"{lag_periods} {cadence_unit}{'s' if lag_periods != 1 else ''}"
    is_significant = p_value is not None and p_value < 0.05

    if strength_word == 'no real':
        headline = f"No real link found between {plain_driver} and malaria cases in this data."
    else:
        direction = 'more' if r_value > 0 else 'fewer'
        headline = (
            f"{plain_driver.capitalize()} has {strength_word} link with malaria cases: periods with higher "
            f"{plain_driver} tend to be followed, about {lag_phrase} later, by {direction} confirmed cases."
        )

    if p_value is None:
        confidence = "Not enough data yet to tell whether this is a real pattern or just chance."
        confidence_tone = 'warning'
    elif is_significant:
        confidence = f"Based on {sample_size} reporting periods, this pattern is unlikely to be a coincidence."
        confidence_tone = 'success'
    else:
        confidence = f"Based on {sample_size} reporting periods, this could still just be chance — not enough evidence yet."
        confidence_tone = 'warning'

    return {
        'headline': headline,
        'confidence': confidence,
        'confidence_tone': confidence_tone,
        'strength_word': strength_word.replace('a ', '').replace('no real', 'No real'),
        'is_significant': is_significant,
    }

_RF_MIN_TRAIN = 20
_RF_MIN_TEST = 5
_RF_FEATURE_NAMES = ['Rainfall a while earlier', 'Temperature a while earlier', 'How many cases there were last time']

# (upper bound, plain-language accuracy word) — how much of the year-to-year
# pattern the model's R² captures, in words instead of a bare statistic.
_RF_ACCURACY_BANDS = [
    (0.0, 'Not yet accurate'),
    (0.3, 'Slightly accurate'),
    (0.6, 'Somewhat accurate'),
    (0.8, 'Fairly accurate'),
    (1.01, 'Highly accurate'),
]


def _describe_forecast_accuracy(r2):
    """Plain-language read on the forecast model's R² — see _RF_ACCURACY_BANDS."""
    if r2 <= 0:
        return 'Not yet accurate'
    return next(word for ceiling, word in _RF_ACCURACY_BANDS if r2 < ceiling)


def _random_forest_forecast(lag_periods):
    """
    Trains a RandomForestRegressor to predict a district's confirmed cases for a
    reporting period from: rainfall and temperature `lag_periods` periods earlier
    (the same offset the correlation panel above uses), plus the immediately prior
    period's own confirmed-case count as an autoregressive "momentum" term.

    District identity is deliberately NOT used as a feature. With only a handful of
    records per district, one-hot-encoding 100+ districts would let the model just
    memorize each district's baseline case level rather than learn a relationship
    that generalizes — the same overfitting trap as fitting a line through 2 points.

    Evaluated with a time-based holdout (train on every period except the most
    recent reporting year, test only on that held-out year) rather than a random
    split, so R²/MAE measure genuine forecasting skill on unseen future data instead
    of in-sample fit, which a small random-forest can trivially inflate.

    Returns a dict with 'available': False (plus 'n_samples') when there isn't
    enough data for a meaningful holdout — never a model trained and evaluated on
    the same rows.
    """
    records = list(
        IntegratedMalariaData.objects
        .order_by('district_id', 'date')
        .values('district_id', 'district__name', 'date', 'reporting_year',
                 'rainfall_mm', 'avg_temperature_c', 'rdt_confirmations')
    )
    by_district = {}
    for r in records:
        by_district.setdefault(r['district_id'], []).append(r)

    samples = []
    for recs in by_district.values():
        for i in range(lag_periods, len(recs)):
            samples.append({
                'features': [
                    recs[i - lag_periods]['rainfall_mm'],
                    recs[i - lag_periods]['avg_temperature_c'],
                    recs[i - 1]['rdt_confirmations'],
                ],
                'target': recs[i]['rdt_confirmations'],
                'year': recs[i]['reporting_year'],
                'district_name': recs[i]['district__name'],
            })

    distinct_years = sorted({s['year'] for s in samples})
    if len(samples) < _RF_MIN_TRAIN + _RF_MIN_TEST or len(distinct_years) < 2:
        return {'available': False, 'n_samples': len(samples)}

    test_year = distinct_years[-1]
    train_samples = [s for s in samples if s['year'] < test_year]
    test_samples = [s for s in samples if s['year'] == test_year]
    if len(train_samples) < _RF_MIN_TRAIN or len(test_samples) < _RF_MIN_TEST:
        return {'available': False, 'n_samples': len(samples)}

    X_train = np.array([s['features'] for s in train_samples])
    y_train = np.array([s['target'] for s in train_samples])
    X_test = np.array([s['features'] for s in test_samples])
    y_test = np.array([s['target'] for s in test_samples])

    # Shallow, leaf-constrained trees on purpose — with only a few hundred rows a
    # deep unconstrained forest memorizes the training set outright.
    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    importance = sorted(
        zip(_RF_FEATURE_NAMES, model.feature_importances_), key=lambda pair: -pair[1]
    )

    # Largest actual burden first — the districts a health team would actually
    # check the model's judgment against first.
    per_district = sorted(
        zip(test_samples, y_test.tolist(), predictions.tolist()),
        key=lambda triple: -triple[1],
    )

    return {
        'available': True,
        'r2': r2_score(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'n_train': len(train_samples),
        'n_test': len(test_samples),
        'test_year': test_year,
        'feature_importance': [(name, round(float(pct) * 100, 1)) for name, pct in importance],
        'actual_vs_predicted': [
            (s['district_name'], int(actual), round(float(pred), 1)) for s, actual, pred in per_district
        ],
    }


@login_required
def analytics_view(request):
    driver = request.GET.get('driver', 'rainfall')
    if driver not in _DRIVER_FIELDS:
        driver = 'rainfall'
    driver_field, driver_label, plain_driver, driver_color = _DRIVER_FIELDS[driver]

    try:
        lag_weeks = int(request.GET.get('lag', 3))
    except (TypeError, ValueError):
        lag_weeks = 3
    lag_weeks = max(1, min(6, lag_weeks))

    records = list(
        IntegratedMalariaData.objects
        .order_by('district_id', 'date')
        .values('district_id', 'date', driver_field, 'rdt_confirmations')
    )

    # Group per district so the lag is applied along each district's own timeline,
    # then correlate the chosen driver at week (i - lag_weeks) against confirmed cases at week i.
    by_district = {}
    for r in records:
        by_district.setdefault(r['district_id'], []).append(r)

    lagged_driver, lagged_cases = [], []
    for recs in by_district.values():
        for i in range(lag_weeks, len(recs)):
            lagged_driver.append(recs[i - lag_weeks][driver_field])
            lagged_cases.append(recs[i]['rdt_confirmations'])

    r_value = _pearson_r(lagged_driver, lagged_cases)
    p_value = _p_value_from_r(r_value, len(lagged_driver))

    # Some imports (e.g. the NMEC/DHIS2 annual export) never carry real weather
    # figures — rainfall_mm/avg_temperature_c are left at their model defaults for
    # every row. A fully-constant driver is uncomputable in principle (zero
    # variance). But a *mixed* database — some rows with real readings, most
    # stamped with the placeholder default from a case-only import — still shows
    # nonzero variance and produces a plausible-looking r/p, even though the result
    # is mostly noise from the placeholder rows diluting a small pocket of real
    # data. So flag on the placeholder share of the actual correlated sample, not
    # just on whether any variance exists at all.
    driver_default = _DRIVER_DEFAULTS.get(driver_field)
    weather_unavailable = len(lagged_driver) >= 2 and len(set(lagged_driver)) <= 1
    placeholder_count = sum(1 for v in lagged_driver if v == driver_default)
    placeholder_fraction = placeholder_count / len(lagged_driver) if lagged_driver else 0
    weather_mostly_placeholder = not weather_unavailable and placeholder_fraction >= 0.5

    # Every reporting period, aggregated nationally, oldest -> newest — the basis
    # for the overlay chart below.
    all_period_rows = list(
        IntegratedMalariaData.objects
        .values('reporting_year', 'epi_week')
        .annotate(driver_avg=Avg(driver_field), cases=Sum('rdt_confirmations'))
        .order_by('reporting_year', 'epi_week')
    )

    # This NMEC-style dataset reports annually (every record sits on the epi_week=1
    # placeholder) rather than weekly — same detection the Dashboard trend uses.
    # Reuse it so the Random Forest section below doesn't call a 1-record lag
    # "1 week" when it's actually a full year.
    is_annual_cadence = len({r['epi_week'] for r in all_period_rows}) <= 1
    cadence_unit = 'year' if is_annual_cadence else 'week'

    # The chart must show the SAME lagged relationship the stats above it
    # describe — the driver value from `lag_weeks` periods earlier plotted
    # against confirmed cases now — not the driver and cases from the same
    # period. Plotting them unlagged would visually contradict a page that's
    # telling the reader "rainfall N periods earlier predicts cases now."
    # Most recent 12 aligned pairs, oldest -> newest.
    weekly_rows = [
        {
            'epi_week': all_period_rows[i]['epi_week'],
            'reporting_year': all_period_rows[i]['reporting_year'],
            'driver_avg': all_period_rows[i - lag_weeks]['driver_avg'],
            'cases': all_period_rows[i]['cases'],
        }
        for i in range(lag_weeks, len(all_period_rows))
    ][-12:]

    # Same annual-vs-weekly label choice the Dashboard trend uses — labeling every
    # point "W1" when the data is actually annual would make every x-axis tick on
    # this chart identical and unreadable.
    if is_annual_cadence:
        chart_labels = [str(r['reporting_year']) for r in weekly_rows]
    else:
        chart_labels = [f"W{r['epi_week']}" for r in weekly_rows]

    plain_summary = _describe_correlation(r_value, p_value, plain_driver, lag_weeks, cadence_unit, len(lagged_driver))

    rf = _random_forest_forecast(lag_weeks)

    lag_phrase = f"{lag_weeks} {cadence_unit}{'s' if lag_weeks != 1 else ''} earlier"

    # ---- Estimated monthly breakdown (folded into this page rather than a
    # separate one — same "annual-only data" context, so it belongs alongside
    # the rest of the annual-cadence handling above, not off on its own). ----
    md_districts = (
        ZambianDistrict.objects.filter(integratedmalariadata__isnull=False).distinct().order_by('name')
    )
    md_district_id = request.GET.get('district_id') or None
    md_selected_district = None
    if md_district_id:
        try:
            md_selected_district = ZambianDistrict.objects.get(pk=md_district_id)
        except (ZambianDistrict.DoesNotExist, ValueError):
            md_selected_district = None
    if not md_selected_district:
        md_selected_district = md_districts.first()

    md_year = None
    md_available_years = []
    md_annual_total = None
    md_monthly_breakdown = []
    if md_selected_district:
        md_available_years = list(
            IntegratedMalariaData.objects
            .filter(district=md_selected_district)
            .values_list('reporting_year', flat=True)
            .distinct()
            .order_by('-reporting_year')
        )
        try:
            md_year = int(request.GET.get('year')) if request.GET.get('year') else None
        except ValueError:
            md_year = None
        if md_year not in md_available_years:
            md_year = md_available_years[0] if md_available_years else None

        if md_year is not None:
            md_annual_total = (
                IntegratedMalariaData.objects
                .filter(district=md_selected_district, reporting_year=md_year)
                .aggregate(total=Sum('rdt_confirmations'))['total'] or 0
            )
            md_estimated = _distribute_annual_to_months(md_annual_total)
            md_monthly_breakdown = [
                {'short': _MONTH_NAMES[m - 1][:3], 'cases': md_estimated[m]} for m in range(1, 13)
            ]

    context = {
        'driver': driver,
        'driver_label': driver_label,
        'driver_label_lagged': f"{driver_label} ({lag_phrase})",
        'plain_driver': plain_driver,
        'driver_color': driver_color,
        'plain_summary': plain_summary,
        'correlation_r': f"{r_value:+.2f}" if r_value is not None else 'N/A',
        'p_value': f"{p_value:.3f}" if p_value is not None else 'N/A',
        'is_significant': p_value is not None and p_value < 0.05,
        'lag_weeks': lag_weeks,
        'lag_options': range(1, 7),
        'sample_size': len(lagged_driver),
        'has_data': len(lagged_driver) >= 4,
        'weather_unavailable': weather_unavailable,
        'weather_mostly_placeholder': weather_mostly_placeholder,
        'placeholder_pct': round(placeholder_fraction * 100),
        'chart_labels_json': json.dumps(chart_labels),
        'chart_driver_json': json.dumps([round(r['driver_avg'] or 0, 1) for r in weekly_rows]),
        'chart_cases_json': json.dumps([r['cases'] or 0 for r in weekly_rows]),
        'cadence_unit': cadence_unit,
        'rf_available': rf['available'],
        'rf_n_samples': rf.get('n_samples'),
        'rf_r2': f"{rf['r2']:.2f}" if rf.get('available') else None,
        'rf_r2_pct': round(rf['r2'] * 100) if rf.get('available') else None,
        'rf_accuracy_word': _describe_forecast_accuracy(rf['r2']) if rf.get('available') else None,
        'rf_mae': f"{rf['mae']:.1f}" if rf.get('available') else None,
        'rf_n_train': rf.get('n_train'),
        'rf_n_test': rf.get('n_test'),
        'rf_test_year': rf.get('test_year'),
        'rf_feature_importance': rf.get('feature_importance'),
        'rf_avp_shown': min(20, len(rf.get('actual_vs_predicted', []))),
        'rf_avp_labels_json': json.dumps([d for d, _a, _p in rf.get('actual_vs_predicted', [])[:20]]),
        'rf_avp_actual_json': json.dumps([a for _d, a, _p in rf.get('actual_vs_predicted', [])[:20]]),
        'rf_avp_predicted_json': json.dumps([p for _d, _a, p in rf.get('actual_vs_predicted', [])[:20]]),
        'md_districts': md_districts,
        'md_selected_district': md_selected_district,
        'md_available_years': md_available_years,
        'md_selected_year': md_year,
        'md_annual_total': md_annual_total,
        'md_monthly_breakdown': md_monthly_breakdown,
        'md_monthly_labels_json': json.dumps([row['short'] for row in md_monthly_breakdown]),
        'md_monthly_cases_json': json.dumps([row['cases'] for row in md_monthly_breakdown]),
    }
    return render(request, 'analytics.html', context)

# ================= 4b. ESTIMATED MONTHLY DISTRIBUTION (NOT REAL DATA) =================
# The uploaded NMEC data reports annually. There is no real monthly breakdown to
# show. This section deliberately never touches IntegratedMalariaData or any of
# the real analytics above — it only ever produces a clearly-labeled, on-the-fly
# ESTIMATE, splitting an annual total across 12 months using a documented
# seasonality curve, for anyone who wants to see/download an approximate monthly
# shape. It must never be presented, stored, or exported as if it were observed
# data — see the "estimated" labeling throughout the view and template.
#
# Curve source: Zambia's malaria transmission consistently peaks Feb-Apr (rainy
# season) and troughs Jul-Oct (dry season), per NMEC/DHIS2-sourced literature:
#   - "Rethinking Malaria Seasonality: Humidity-Driven Transmission Shifts in
#     Zambia (2009-2023)" (medRxiv 2025) - incidence highest Feb-Apr, lowest Jul-Oct
#   - "Impact of aerial humidity on seasonal malaria: an ecological study in
#     Zambia" (Malaria Journal, 2022) - rainy season Nov-Apr driving transmission
# Neither paper publishes an exact numeric monthly index, so these are relative
# weights approximating the described shape, not a published table — normalized
# to sum to 1 in code below. Treat the resulting monthly split as illustrative of
# the general pattern, not a precise reconstruction of any specific year.
_MONTHLY_SEASONALITY_WEIGHTS = {
    1: 9.0, 2: 11.0, 3: 14.0, 4: 13.0, 5: 9.0, 6: 5.0,
    7: 3.0, 8: 2.5, 9: 3.0, 10: 4.0, 11: 6.0, 12: 9.5,
}
_MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
]


def _distribute_annual_to_months(annual_total, weights=_MONTHLY_SEASONALITY_WEIGHTS):
    """
    Splits an integer annual total across 12 months by the given relative
    weights, using the largest-remainder method so the 12 outputs are integers
    that sum EXACTLY back to annual_total (a naive round() per month can drift
    the total by a case or two, which would be a self-inflicted "why don't
    these add up" credibility problem for numbers that are already labeled as
    estimates).
    """
    total_weight = sum(weights.values())
    raw = {m: annual_total * w / total_weight for m, w in weights.items()}
    floors = {m: int(v) for m, v in raw.items()}
    remainder = annual_total - sum(floors.values())
    # Hand out the leftover units to the months with the largest fractional
    # remainder first — the standard largest-remainder apportionment method.
    by_remainder = sorted(raw.keys(), key=lambda m: raw[m] - floors[m], reverse=True)
    for m in by_remainder[:remainder]:
        floors[m] += 1
    return floors  # {month_number: estimated_cases}


@login_required
def monthly_estimate_csv_view(request):
    """Downloadable CSV of the estimate — headers make clear it's derived, not observed."""
    district_id = request.GET.get('district_id')
    year = request.GET.get('year')
    district = ZambianDistrict.objects.filter(pk=district_id).first()
    if not district or not year:
        messages.error(request, "Pick a district and year first.")
        return redirect('monthly_estimate')

    annual_total = (
        IntegratedMalariaData.objects
        .filter(district=district, reporting_year=year)
        .aggregate(total=Sum('rdt_confirmations'))['total'] or 0
    )
    estimated = _distribute_annual_to_months(annual_total)

    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="{district.name}_{year}_monthly_ESTIMATE.csv"'
    writer = csv.writer(response)
    writer.writerow(['# ESTIMATED DATA - NOT OBSERVED', f'{district.name} does not have real monthly NMEC records for {year}.'])
    writer.writerow(['# This file splits the real annual total below across 12 months using a published Zambia'])
    writer.writerow(['# malaria seasonality curve (peak Feb-Apr, trough Jul-Oct) - it is a modeled approximation,'])
    writer.writerow(['# not a measurement. Do not present these monthly figures as real NMEC data.'])
    writer.writerow([f'# Real annual total ({year}, confirmed RDT cases): {annual_total}'])
    writer.writerow([])
    writer.writerow(['district', 'year', 'month', 'estimated_rdt_confirmations'])
    for m in range(1, 13):
        writer.writerow([district.name, year, _MONTH_NAMES[m - 1], estimated[m]])
    return response

# ================= 4c. DATA QUALITY: COMPLETENESS & TIMELINESS =================
# Two things this honestly can and can't measure, given the actual schema:
#   - Reporting completeness (which periods a district DID/DIDN'T report) and a
#     staleness proxy (is the district's latest report current with the rest of
#     the dataset) - both fully computable from what's on file.
#   - "Facility-level" anything, and true DHIS2-style timeliness (days late
#     against a submission deadline) - NOT computable. IntegratedMalariaData is
#     district-level and has no submission-timestamp field, only the reporting
#     period itself. The page says so rather than quietly faking either one.
@login_required
def data_quality_view(request):
    all_periods = list(
        IntegratedMalariaData.objects.values_list('reporting_year', 'epi_week').distinct()
    )
    expected_periods = set(all_periods)
    n_expected = len(expected_periods)
    is_annual_cadence = len({p[1] for p in all_periods}) <= 1
    cadence_unit = 'year' if is_annual_cadence else 'week'
    latest_period = max(expected_periods) if expected_periods else None

    def period_label(year, week):
        return str(year) if is_annual_cadence else f"W{week} '{str(year)[-2:]}"

    rows = []
    for d in ZambianDistrict.objects.all().order_by('name'):
        reported_periods = set(
            IntegratedMalariaData.objects.filter(district=d).values_list('reporting_year', 'epi_week')
        )
        missing = sorted(expected_periods - reported_periods)
        completeness = round(len(reported_periods) / n_expected * 100, 1) if n_expected else 0.0

        if reported_periods:
            district_latest = max(reported_periods)
            is_current = district_latest == latest_period
            latest_label = period_label(*district_latest)
        else:
            is_current = False
            latest_label = 'Never reported'

        rows.append({
            'id': d.id,
            'name': d.name,
            'reported': len(reported_periods),
            'expected': n_expected,
            'completeness': completeness,
            'missing_labels': [period_label(y, w) for y, w in missing],
            'missing_count': len(missing),
            'is_current': is_current,
            'latest_label': latest_label,
        })

    # Worst completeness first — the districts that most need attention surface
    # immediately rather than needing to be found in an alphabetical list.
    rows.sort(key=lambda r: (r['completeness'], r['name']))

    context = {
        'cadence_unit': cadence_unit,
        'n_expected_periods': n_expected,
        'rows': rows,
        'avg_completeness': round(sum(r['completeness'] for r in rows) / len(rows), 1) if rows else None,
        'districts_with_gaps': sum(1 for r in rows if r['missing_count']),
        'districts_never_reported': sum(1 for r in rows if r['reported'] == 0),
        'districts_current': sum(1 for r in rows if r['is_current']),
        'total_districts': len(rows),
    }
    return render(request, 'data_quality.html', context)

# ================= 5. CLINICAL DECISION SUPPORT TIERS =================
_BURDEN_WINDOW = 4


def _recent_case_burden(district):
    """
    Case burden over a rolling window of the district's most recent reporting
    periods (not a single latest date — one week's count is too small/noisy to
    classify against the same incidence thresholds the Dashboard uses on
    cumulative totals). Returns (recent_cases, prior_cases_or_None, reference_date,
    current_window_len, burden_label, burden_badge, incidence_or_None,
    prior_window_len).

    prior_cases is only meaningful as a like-for-like comparison against
    recent_cases when prior_window_len == _BURDEN_WINDOW — with fewer total
    reporting periods on file than 2x the window (true for every district in
    an annual dataset with only ~5 years on record), the "prior window" is
    whatever's left over, which can be as little as a single period. Comparing
    a genuine _BURDEN_WINDOW-period sum against a partial 1-period leftover and
    calling both "N-period windows" would wildly overstate the swing (e.g. 4
    years of cumulative cases vs. a single older year, mislabeled as
    comparable). Callers must check prior_window_len before trusting the
    rising/falling comparison, not just whether prior_cases is None.
    """
    records = list(
        IntegratedMalariaData.objects
        .filter(district=district)
        .order_by('-date')
        .values('date', 'rdt_confirmations')[:_BURDEN_WINDOW * 2]
    )
    reference_date = records[0]['date'] if records else None
    current_window = records[:_BURDEN_WINDOW]
    prior_window = records[_BURDEN_WINDOW:_BURDEN_WINDOW * 2]

    recent_cases = sum(r['rdt_confirmations'] for r in current_window)
    prior_cases = sum(r['rdt_confirmations'] for r in prior_window) if prior_window else None

    burden_label, burden_badge, incidence = _classify_burden(recent_cases, district.population)
    return (
        recent_cases, prior_cases, reference_date, len(current_window),
        burden_label, burden_badge, incidence, len(prior_window),
    )


def _district_priority_queue():
    """
    Ranks every district that has case data by combined urgency — the SAME
    forecast + case-burden decision engine used in the detail panel below — so
    response teams get a single "go here first" list instead of checking
    districts one at a time. Sorted by combined tier severity (critical worst),
    then by confirmed case count as a tiebreaker.
    """
    districts = ZambianDistrict.objects.filter(integratedmalariadata__isnull=False).distinct()

    # Pass 1: case burden is pure DB lookup — cheap, do it sequentially and drop
    # any district with no reporting history yet.
    burden_by_id = {}
    for d in districts:
        recent_cases, _prior_cases, reference_date, _window_len, _burden_label, burden_badge, incidence, _prior_window_len = _recent_case_burden(d)
        if reference_date is None:
            continue
        burden_by_id[d.id] = {
            'district': d,
            'recent_cases': recent_cases,
            'reference_date': reference_date,
            'burden_badge': burden_badge,
            'incidence': incidence,
        }

    # Pass 2: the forecast fetch is the slow part — one network round-trip per
    # distinct location (each individually cached 6h, see _fetch_forecast_days).
    # Fetching them one district at a time serially means a cold cache pays for
    # N round-trips back-to-back on a single page load. Fan them out on a small
    # thread pool instead — network-bound I/O, so threading (not multiprocessing)
    # is enough — and dedupe by rounded coordinate so districts sharing a location
    # only trigger one request.
    coords = {
        (round(info['district'].latitude, 2), round(info['district'].longitude, 2))
        for info in burden_by_id.values()
    }
    forecast_by_coord = {}
    if coords:
        with ThreadPoolExecutor(max_workers=min(10, len(coords))) as executor:
            future_to_coord = {executor.submit(_fetch_forecast_days, lat, lon): (lat, lon) for lat, lon in coords}
            for future, coord in future_to_coord.items():
                forecast_by_coord[coord] = future.result()

    queue = []
    for info in burden_by_id.values():
        d = info['district']
        burden_rank = _RISK_RANK[info['burden_badge']]

        forecast_days = forecast_by_coord.get((round(d.latitude, 2), round(d.longitude, 2)))
        forecast_summary = _forecast_risk_summary(forecast_days)
        forecast_rank = _RISK_RANK[forecast_summary['badge']] if forecast_summary else None

        tier = _combine_decision(forecast_rank, burden_rank)
        meta = _TIER_META[tier]

        queue.append({
            'id': d.id,
            'name': d.name,
            'tier': tier,
            'badge': meta['risk_badge'],
            'tier_label': meta['risk_level'],
            'icon': meta['icon'],
            'cases': info['recent_cases'],
            'incidence': info['incidence'],
            'forecast_available': forecast_summary is not None,
            'date': info['reference_date'],
        })

    # Fixed priority order for the 5 named tiers (critical worst, stable best) rather
    # than raw numeric rank, since "preemptive" and "active" both derive from a max
    # factor rank of 2 but shouldn't tie arbitrarily against each other.
    tier_order = {'critical': 0, 'active': 1, 'preemptive': 2, 'watch': 3, 'stable': 4}
    queue.sort(key=lambda q: (tier_order[q['tier']], -q['cases']))
    return queue


@login_required
def decision_view(request):
    districts_with_data = (
        ZambianDistrict.objects
        .filter(integratedmalariadata__isnull=False)
        .distinct()
        .order_by('name')
    )
    priority_queue = _district_priority_queue()

    district_id = request.GET.get('district_id') or None
    selected_district = None
    if district_id:
        try:
            selected_district = ZambianDistrict.objects.get(pk=district_id)
        except (ZambianDistrict.DoesNotExist, ValueError):
            selected_district = None

    auto_selected = False
    if not selected_district and priority_queue:
        # No explicit choice — default to the top of the priority queue rather than a
        # nationwide average, since forecasts are inherently location-specific.
        selected_district = ZambianDistrict.objects.filter(pk=priority_queue[0]['id']).first()
        auto_selected = True

    if not selected_district:
        return render(request, 'decision.html', {
            'has_data': False,
            'districts_with_data': districts_with_data,
            'selected_district_id': '',
            'priority_queue': [],
            'target_label': 'No Data Yet',
        })

    target_label = selected_district.name

    # Case burden over a rolling window (not a single latest date — see
    # _recent_case_burden's docstring); same helper the priority queue uses above,
    # so the queue and this detail panel always agree on the same district's tier.
    recent_cases, prior_cases, reference_date, window_len, burden_label, burden_badge, incidence, prior_window_len = _recent_case_burden(selected_district)
    burden_rank = _RISK_RANK[burden_badge]

    # Only a genuine like-for-like comparison when the prior window is fully
    # populated with _BURDEN_WINDOW periods too — a partial leftover window
    # (e.g. just 1 period, the norm with only ~5 years on record per district)
    # isn't comparable to a full _BURDEN_WINDOW-period sum and would overstate
    # the swing (a 4-year cumulative total vs. a single older year, both
    # mislabeled as "4-period windows").
    if prior_cases is None or prior_window_len < _BURDEN_WINDOW:
        trend_text, trend_icon, trend_class = 'Not enough reporting history yet for a fair period-over-period comparison', '', 'text-muted'
    elif recent_cases > prior_cases:
        trend_text = f'Rising vs prior {_BURDEN_WINDOW}-period window ({prior_cases} → {recent_cases} cases)'
        trend_icon, trend_class = '▲', 'text-danger'
    elif recent_cases < prior_cases:
        trend_text = f'Falling vs prior {_BURDEN_WINDOW}-period window ({prior_cases} → {recent_cases} cases)'
        trend_icon, trend_class = '▼', 'text-success'
    else:
        trend_text, trend_icon, trend_class = f'Unchanged vs prior {_BURDEN_WINDOW}-period window', '▬', 'text-muted'

    # --- Forecast: next 14 days for this district's coordinates ---
    forecast_days = _fetch_forecast_days(selected_district.latitude, selected_district.longitude)
    forecast_summary = _forecast_risk_summary(forecast_days)
    forecast_rank = _RISK_RANK[forecast_summary['badge']] if forecast_summary else None
    danger_days_7 = forecast_summary['danger_days_7'] if forecast_summary else 0

    # --- Combine both factors into one of 5 tiers, then write the recommendation to match ---
    tier = _combine_decision(forecast_rank, burden_rank)
    meta = _TIER_META[tier]
    risk_badge, risk_level = meta['risk_badge'], meta['risk_level']

    if tier == 'critical':
        recommended_action = (
            f'Both the 7-day forecast ({danger_days_7} high-risk day(s)) and recent confirmed case burden '
            f'({recent_cases} cases over the last {window_len} reporting period(s), {burden_label.lower()}) '
            f'point to compounding risk in {target_label} — spray, distribute nets, and staff up for treatment immediately.'
        )
        checklist = [
            (f'Spray — IRS in {target_label}', 'Dispatch full-coverage indoor residual spraying teams to every household within 24–48h.', f'{danger_days_7} of the next 7 forecast days meet critical breeding thresholds, compounding an already high case count.', 'danger'),
            (f'Distribute Nets (LLINs) in {target_label}', 'House-to-house distribution of long-lasting insecticidal nets to every household.', 'Prioritize under-5 and pregnant-women households first, given the active outbreak.', 'danger'),
            ('Clinical Surge Readiness', 'Verify RDT and antimalarial (ACT) stock levels and staffing at every facility.', f'{recent_cases} confirmed cases in the last {window_len} reporting period(s) means treatment demand is already elevated.', 'danger'),
            ('Larval Source Management', 'Identify and treat standing-water mosquito breeding sites near population centers.', 'Ahead of the forecast wet window — cuts the next generation of mosquitoes before they emerge.', 'warning'),
        ]
    elif tier == 'active':
        recommended_action = (
            f'Recent confirmed case burden in {target_label} is {burden_label.lower()} ({recent_cases} cases over the '
            f'last {window_len} reporting period(s)) — even with a calmer forecast, an active outbreak still needs '
            f'spraying and net distribution to cut transmission, alongside case management.'
        )
        checklist = [
            (f'Spray — IRS in {target_label}', 'Deploy indoor residual spraying teams to the affected areas now.', 'An active outbreak is already underway and needs containing, even though the forecast is calmer.', 'danger'),
            (f'Distribute Nets (LLINs) in {target_label}', 'Distribute long-lasting insecticidal nets to affected households.', 'Reduces ongoing transmission while the outbreak is brought under control.', 'danger'),
            ('Case Management Surge', 'Verify treatment capacity (RDT/ACT stock, staffing) at facilities serving the district.', f'{recent_cases} confirmed cases in the last {window_len} reporting period(s) — demand is already elevated.', 'danger'),
            ('Community Case Surveillance', 'Increase active case-finding in the community.', 'Catches cases the routine reporting system might miss, containing spread.', 'warning'),
        ]
    elif tier == 'preemptive':
        recommended_action = (
            f'The 7-day forecast for {target_label} shows {danger_days_7} high-risk breeding day(s) while recent confirmed '
            f'cases remain {burden_label.lower()} ({recent_cases} over the last {window_len} reporting period(s)) — '
            f'spray and distribute nets now, before cases spike.'
        )
        checklist = [
            (f'Spray — Pre-Emptive IRS in {target_label}', 'Spray ahead of the forecast wet window, before breeding conditions peak.', f'{danger_days_7} of the next 7 forecast days meet critical breeding thresholds.', 'warning'),
            (f'Distribute Nets (LLINs) in {target_label}', 'Top up and distribute long-lasting insecticidal nets to households.', 'Cases are still low — this is prevention, getting ahead of the forecast before transmission rises.', 'warning'),
            ('Baseline Surveillance', 'Maintain the standard weekly case and rainfall monitoring cadence.', f'Only {recent_cases} confirmed cases in the last {window_len} reporting period(s) — no active outbreak yet.', 'secondary'),
        ]
    elif tier == 'watch':
        recommended_action = f'Forecast and case data for {target_label} are both moderate — increase monitoring frequency without full deployment.'
        checklist = [
            ('Increase Monitoring Frequency', 'Move from standard to twice-weekly case and rainfall review.', 'Forecast and case data are both moderate — not urgent yet, but worth watching more closely.', 'warning'),
            ('Stock Readiness Check', 'Confirm IRS/LLIN reserves are positioned for rapid deployment.', 'So the district can respond quickly if conditions worsen.', 'secondary'),
        ]
    else:
        recommended_action = f'Forecast and confirmed case data are both calm for {target_label} — maintain routine surveillance.'
        checklist = [
            ('Routine Surveillance', 'Continue the standard weekly reporting cadence.', 'Forecast and confirmed case data are both calm right now.', 'success'),
            ('Preventive Maintenance', 'Service existing IRS/LLIN stock.', 'Keeps readiness up during the quiet period.', 'secondary'),
        ]

    context = {
        'risk_level': risk_level,
        'recommended_action': recommended_action,
        'risk_badge': risk_badge,
        'tier_icon': meta['icon'],
        'checklist': checklist,
        'recent_cases': recent_cases,
        'burden_label': burden_label,
        'incidence_per_10k': incidence,
        'reference_date': reference_date,
        'has_data': reference_date is not None,
        'target_label': target_label,
        'auto_selected': auto_selected,
        'districts_with_data': districts_with_data,
        'selected_district_id': selected_district.id,
        'trend_text': trend_text,
        'trend_icon': trend_icon,
        'trend_class': trend_class,
        'priority_queue': priority_queue,
        'forecast_available': forecast_summary is not None,
        'forecast_preview': forecast_days[:7] if forecast_days else [],
        'danger_days_7': danger_days_7,
        'warning_days_7': forecast_summary['warning_days_7'] if forecast_summary else 0,
    }
    return render(request, 'decision.html', context)

# ================= 6. METEOROLOGICAL VIEW: SATELLITE API LINK =================
@login_required
def weather_view(request):
    all_districts = ZambianDistrict.objects.all().order_by('name')

    selected_district = None
    district_id = request.GET.get('district_id') or None
    if district_id:
        try:
            selected_district = all_districts.get(pk=district_id)
        except (ZambianDistrict.DoesNotExist, ValueError):
            selected_district = None

    if selected_district:
        target_lat, target_lon, location_label = selected_district.latitude, selected_district.longitude, selected_district.name
    else:
        target_lat, target_lon, location_label = -15.42, 28.28, 'Lusaka (Default)'

    forecast_days = _fetch_forecast_days(target_lat, target_lon)

    if forecast_days:
        forecast_cards = []
        for day in forecast_days:
            risk_status, _badge = _classify_risk(day['rain'], day['temp'])
            parsed_day = datetime.datetime.strptime(day['date'], "%Y-%m-%d")
            forecast_cards.append({
                'day': parsed_day.strftime("%A"),
                'date': parsed_day.strftime("%d %b"),
                'rain': f"{day['rain']:.1f} mm",
                'temp': f"{day['temp']:.1f}°C",
                'risk': risk_status,
                'badge': day['badge']
            })
    else:
        forecast_cards = [
            {'day': 'Forecast Unavailable', 'date': '', 'rain': '—', 'temp': '—', 'risk': 'Weather API unreachable — try again shortly', 'badge': 'secondary'},
        ]

    return render(request, 'weather.html', {
        'forecasts': forecast_cards,
        'all_districts': all_districts,
        'selected_district_id': selected_district.id if selected_district else '',
        'location_label': location_label,
    })

# ================= 7. DATA UPLOAD PORT: CSV HEADER MAPPER =================
SAMPLE_CSV_HEADER = ['district', 'date', 'epi_week', 'reporting_year', 'rdt_confirmations']
SAMPLE_CSV_ROWS = [
    ['Chadiza', '2026-01-05', '1', '2026', '68'],
    ['Chadiza', '2026-01-12', '2', '2026', '81'],
    ['Lusaka', '2026-01-05', '1', '2026', '150'],
    ['Lusaka', '2026-01-12', '2', '2026', '142'],
]


@login_required
def sample_csv_view(request):
    """Serves a downloadable sample CSV matching the columns the upload parser expects."""
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="sample_malaria_data.csv"'
    writer = csv.writer(response)
    writer.writerow(SAMPLE_CSV_HEADER)
    writer.writerows(SAMPLE_CSV_ROWS)
    return response


@login_required
def upload_view(request):
    if request.method == 'POST' and request.FILES.get('malaria_file'):
        csv_file = request.FILES['malaria_file']

        if not csv_file.name.endswith('.csv'):
            messages.error(request, "Invalid file format. Please upload a standard CSV spreadsheet.")
            return render(request, 'upload.html')

        try:
            data_set = csv_file.read().decode('UTF-8')
            io_string = io.StringIO(data_set)

            reader = csv.reader(io_string, delimiter=',')
            headers = [h.strip().lower() for h in next(reader)]

            # --- INTELLIGENT COLUMN AUTOMATIC DETECTOR ---
            def find_index(keywords, default_idx):
                for i, h in enumerate(headers):
                    if any(k in h for k in keywords):
                        return i
                return default_idx

            def find_optional_index(keywords):
                """Like find_index, but returns None (rather than guessing a column) when absent."""
                for i, h in enumerate(headers):
                    if any(k in h for k in keywords):
                        return i
                return None

            dist_idx = find_index(['dist', 'location', 'area'], 0)
            date_idx = find_index(['date', 'time', 'period'], 1)
            week_idx = find_index(['week', 'epi'], 2)
            year_idx = find_index(['year', 'yr'], 3)
            rdt_idx  = find_index(['rdt', 'confirm', 'pos'], 4)
            pop_idx  = find_optional_index(['pop'])  # optional — enables incidence-rate hotspot tiers

            def safe_int(value_str):
                try:
                    return int(float(str(value_str).strip()))
                except (ValueError, TypeError):
                    return 0

            created_count = 0
            updated_count = 0
            for row in reader:
                if not row or len(row) <= max(dist_idx, date_idx, week_idx, year_idx, rdt_idx):
                    continue

                district_name = row[dist_idx].strip()
                record_date_str = row[date_idx].strip()
                epi_week      = safe_int(row[week_idx])
                reporting_year = safe_int(row[year_idx])
                rdt_positives = safe_int(row[rdt_idx])

                if not district_name:
                    continue

                # 'Unknown' rather than a name that could pass for a real province
                # (e.g. an earlier version used 'Surveillance Region') — anything
                # that reads as plausible would silently show up as a fake province
                # in any province-level rollup (Dashboard, reports) until someone
                # noticed and corrected it by hand.
                district_obj, _district_created = ZambianDistrict.objects.get_or_create(
                    name=district_name,
                    defaults={'province': 'Unknown', 'latitude': -15.42, 'longitude': 28.28}
                )

                if pop_idx is not None and pop_idx < len(row):
                    pop_val = safe_int(row[pop_idx])
                    if pop_val > 0 and district_obj.population != pop_val:
                        district_obj.population = pop_val
                        district_obj.save(update_fields=['population'])

                try:
                    parsed_date = datetime.datetime.strptime(record_date_str, "%Y-%m-%d").date()
                except ValueError:
                    parsed_date = datetime.date.today()

                _record, was_created = IntegratedMalariaData.objects.update_or_create(
                    district=district_obj,
                    date=parsed_date,
                    defaults={
                        'epi_week': epi_week,
                        'reporting_year': reporting_year,
                        'rdt_confirmations': rdt_positives
                    }
                )
                if was_created:
                    created_count += 1
                else:
                    updated_count += 1

            messages.success(
                request,
                f"Data pipeline successful: {created_count} new observation(s) added, "
                f"{updated_count} existing observation(s) updated."
            )

        except Exception:
            logger.exception("Failed to parse uploaded CSV %r", csv_file.name)
            messages.error(
                request,
                "Failed to parse spreadsheet file structure. Check that the file is a valid CSV with the "
                "expected columns and try again."
            )

    return render(request, 'upload.html')

# ================= 8. ADMINISTRATIVE USER CREATION CONTROL =================
@login_required
def users_view(request):
    if request.user.role != 'ADMIN':
        messages.error(request, "Unauthorized Access Blocked. Administrative Privileges Required.")
        return redirect('dashboard')

    if request.method == 'POST':
        form_action = request.POST.get('form_action')

        if form_action == 'create':
            u = (request.POST.get('new_username') or '').strip()
            r = request.POST.get('new_role')
            p = request.POST.get('new_password') or ''
            p_confirm = request.POST.get('new_password_confirm') or ''
            district = (request.POST.get('new_district') or '').strip()

            if p != p_confirm:
                messages.error(request, "Password and confirmation do not match.")
            else:
                try:
                    validate_password(p)
                except ValidationError as e:
                    for err in e.messages:
                        messages.error(request, err)
                else:
                    try:
                        SystemUser.objects.create_user(
                            username=u, role=r, password=p, district_assignment=district or None
                        )
                        messages.success(request, f"New profile ({u}) provisioned successfully.")
                    except IntegrityError:
                        messages.error(request, f"Username '{u}' is already taken. Choose a different handle.")

        elif form_action == 'toggle':
            target_id = request.POST.get('user_id')
            target_user = SystemUser.objects.filter(pk=target_id).first()
            if target_user and target_user != request.user:
                target_user.is_active = not target_user.is_active
                target_user.save()
                state = "activated" if target_user.is_active else "deactivated"
                messages.success(request, f"Account '{target_user.username}' {state}.")
            else:
                messages.error(request, "Cannot modify that account.")

        elif form_action == 'reset_password':
            target_id = request.POST.get('user_id')
            new_pw = request.POST.get('reset_new_password') or ''
            new_pw_confirm = request.POST.get('reset_new_password_confirm') or ''
            target_user = SystemUser.objects.filter(pk=target_id).first()
            if not target_user:
                messages.error(request, "Account not found.")
            elif new_pw != new_pw_confirm:
                messages.error(request, "New password and confirmation do not match.")
            else:
                try:
                    validate_password(new_pw, user=target_user)
                except ValidationError as e:
                    for err in e.messages:
                        messages.error(request, err)
                else:
                    target_user.set_password(new_pw)
                    target_user.save()
                    messages.success(request, f"Password reset for '{target_user.username}'.")

    all_users = SystemUser.objects.all().order_by('username')
    return render(request, 'users.html', {'system_users': all_users})
