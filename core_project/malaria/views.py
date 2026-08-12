import csv
import io
import json
import math
import datetime
import requests
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.password_validation import validate_password
from django.core.exceptions import ValidationError
from django.core.cache import cache
from django.db import IntegrityError
from django.db.models import Sum, Avg, Max
from django.http import HttpResponse
from .models import SystemUser, IntegratedMalariaData, ZambianDistrict

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
        total_suspected=Sum('suspected_cases'),
        total_confirmed=Sum('rdt_confirmations'),
        latest_date=Max('date'),
    )
    total_suspected = totals['total_suspected'] or 0
    total_confirmed = totals['total_confirmed'] or 0
    positivity_rate = round((total_confirmed / total_suspected) * 100, 1) if total_suspected else 0.0

    # Roll up every district that has records, for both the "critical node"
    # headline stat and the Leaflet hotspot map.
    district_rollup = (
        IntegratedMalariaData.objects
        .values('district__name', 'district__latitude', 'district__longitude', 'district__population')
        .annotate(
            cases=Sum('rdt_confirmations'),
            suspected=Sum('suspected_cases'),
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
            'name': row['district__name'],
            'lat': row['district__latitude'],
            'lon': row['district__longitude'],
            'cases': cases,
            'suspected': row['suspected'] or 0,
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

    # Last 8 reported epidemiological weeks, oldest -> newest, for the trend strip.
    trend_rows = list(
        IntegratedMalariaData.objects
        .values('epi_week', 'reporting_year')
        .annotate(cases=Sum('rdt_confirmations'))
        .order_by('-reporting_year', '-epi_week')[:8]
    )
    trend_rows.reverse()

    context = {
        'total_suspected': total_suspected,
        'confirmed_rdt': total_confirmed,
        'positivity_rate': positivity_rate,
        'alert_district': alert_district,
        'active_user_role': request.user.get_role_display(),
        'has_data': total_suspected > 0,
        'latest_date': totals['latest_date'],
        'map_points': map_points,
        'map_points_json': json.dumps(map_points),
        'trend_labels_json': json.dumps([f"W{r['epi_week']}" for r in trend_rows]),
        'trend_cases_json': json.dumps([r['cases'] or 0 for r in trend_rows]),
    }
    return render(request, 'dashboard.html', context)

# ================= 4. ANALYTICS CORRELATION ENGINE VIEW =================
_DRIVER_FIELDS = {
    'rainfall': ('rainfall_mm', 'Rainfall (mm)', '#0d6efd'),
    'temperature': ('avg_temperature_c', 'Avg. Temperature (°C)', '#f59e0b'),
}

@login_required
def analytics_view(request):
    driver = request.GET.get('driver', 'rainfall')
    if driver not in _DRIVER_FIELDS:
        driver = 'rainfall'
    driver_field, driver_label, driver_color = _DRIVER_FIELDS[driver]

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

    # Most recent 12 reported weeks, aggregated nationally, for the overlay chart.
    weekly_rows = list(
        IntegratedMalariaData.objects
        .values('reporting_year', 'epi_week')
        .annotate(driver_avg=Avg(driver_field), cases=Sum('rdt_confirmations'))
        .order_by('-reporting_year', '-epi_week')[:12]
    )
    weekly_rows.reverse()

    context = {
        'driver': driver,
        'driver_label': driver_label,
        'driver_color': driver_color,
        'correlation_r': f"{r_value:+.2f}" if r_value is not None else 'N/A',
        'p_value': f"{p_value:.3f}" if p_value is not None else 'N/A',
        'is_significant': p_value is not None and p_value < 0.05,
        'lag_weeks': lag_weeks,
        'lag_options': range(1, 7),
        'sample_size': len(lagged_driver),
        'has_data': len(lagged_driver) >= 4,
        'chart_labels_json': json.dumps([f"W{r['epi_week']}" for r in weekly_rows]),
        'chart_driver_json': json.dumps([round(r['driver_avg'] or 0, 1) for r in weekly_rows]),
        'chart_cases_json': json.dumps([r['cases'] or 0 for r in weekly_rows]),
    }
    return render(request, 'analytics.html', context)

# ================= 5. CLINICAL DECISION SUPPORT TIERS =================
_BURDEN_WINDOW = 4


def _recent_case_burden(district):
    """
    Case burden over a rolling window of the district's most recent reporting
    periods (not a single latest date — one week's count is too small/noisy to
    classify against the same incidence thresholds the Dashboard uses on
    cumulative totals). Returns (recent_cases, prior_cases_or_None, reference_date,
    current_window_len, burden_label, burden_badge, incidence_or_None).
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
    return recent_cases, prior_cases, reference_date, len(current_window), burden_label, burden_badge, incidence


def _district_priority_queue():
    """
    Ranks every district that has case data by combined urgency — the SAME
    forecast + case-burden decision engine used in the detail panel below — so
    response teams get a single "go here first" list instead of checking
    districts one at a time. Sorted by combined tier severity (critical worst),
    then by confirmed case count as a tiebreaker.
    """
    districts = ZambianDistrict.objects.filter(integratedmalariadata__isnull=False).distinct()
    queue = []
    for d in districts:
        recent_cases, _prior_cases, reference_date, _window_len, _burden_label, burden_badge, incidence = _recent_case_burden(d)
        if reference_date is None:
            continue
        burden_rank = _RISK_RANK[burden_badge]

        forecast_days = _fetch_forecast_days(d.latitude, d.longitude)
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
            'cases': recent_cases,
            'incidence': incidence,
            'forecast_available': forecast_summary is not None,
            'date': reference_date,
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
    recent_cases, prior_cases, reference_date, window_len, burden_label, burden_badge, incidence = _recent_case_burden(selected_district)
    burden_rank = _RISK_RANK[burden_badge]

    if prior_cases is None:
        trend_text, trend_icon, trend_class = 'Not enough reporting history yet for a period-over-period trend', '', 'text-muted'
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
            (f'Spray — Indoor Residual Spraying (IRS) in {target_label}', f'Dispatch full-coverage spraying within 24–48h; {danger_days_7} of the next 7 forecast days meet critical breeding thresholds', 'danger'),
            (f'Distribute Mosquito Nets (LLINs) in {target_label}', 'House-to-house distribution to every household, prioritizing under-5 and pregnant-women households', 'danger'),
            ('Clinical Surge Readiness', f'{recent_cases} confirmed cases in the last {window_len} reporting period(s) — verify RDT/ACT stock and staffing', 'danger'),
            ('Larval Source Management', 'Target standing-water breeding sites identified ahead of the forecast wet window', 'warning'),
        ]
    elif tier == 'active':
        recommended_action = (
            f'Recent confirmed case burden in {target_label} is {burden_label.lower()} ({recent_cases} cases over the '
            f'last {window_len} reporting period(s)) — even with a calmer forecast, an active outbreak still needs '
            f'spraying and net distribution to cut transmission, alongside case management.'
        )
        checklist = [
            (f'Spray — Indoor Residual Spraying (IRS) in {target_label}', 'Deploy spraying to affected areas now to contain the ongoing outbreak', 'danger'),
            (f'Distribute Mosquito Nets (LLINs) in {target_label}', 'Distribute nets to affected households to reduce ongoing transmission', 'danger'),
            ('Case Management Surge', f'{recent_cases} confirmed cases in the last {window_len} reporting period(s) — verify treatment capacity', 'danger'),
            ('Community Case Surveillance', 'Increase active case-finding to contain spread', 'warning'),
        ]
    elif tier == 'preemptive':
        recommended_action = (
            f'The 7-day forecast for {target_label} shows {danger_days_7} high-risk breeding day(s) while recent confirmed '
            f'cases remain {burden_label.lower()} ({recent_cases} over the last {window_len} reporting period(s)) — '
            f'spray and distribute nets now, before cases spike.'
        )
        checklist = [
            (f'Spray — Pre-Emptive IRS in {target_label}', f'{danger_days_7} of the next 7 forecast days meet critical breeding thresholds — spray ahead of them', 'warning'),
            (f'Distribute Mosquito Nets (LLINs) in {target_label}', 'Top up and distribute nets ahead of the forecast wet window, before transmission rises', 'warning'),
            ('Baseline Surveillance', f'Only {recent_cases} confirmed cases in the last {window_len} reporting period(s) — maintain standard monitoring while prevention deploys', 'secondary'),
        ]
    elif tier == 'watch':
        recommended_action = f'Forecast and case data for {target_label} are both moderate — increase monitoring frequency without full deployment.'
        checklist = [
            ('Increase Monitoring Frequency', 'Move to twice-weekly case and rainfall review', 'warning'),
            ('Stock Readiness Check', 'Confirm IRS/LLIN reserves are positioned for rapid deployment', 'secondary'),
        ]
    else:
        recommended_action = f'Forecast and confirmed case data are both calm for {target_label} — maintain routine surveillance.'
        checklist = [
            ('Routine Surveillance', 'Continue standard weekly reporting cadence', 'success'),
            ('Preventive Maintenance', 'Service existing IRS/LLIN stock for readiness', 'secondary'),
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
            {'day': 'Sync Idle', 'date': '', 'rain': '0.0 mm', 'temp': '25.0°C', 'risk': 'Cache Mode Active', 'badge': 'secondary'},
        ]

    return render(request, 'weather.html', {
        'forecasts': forecast_cards,
        'all_districts': all_districts,
        'selected_district_id': selected_district.id if selected_district else '',
        'location_label': location_label,
    })

# ================= 7. DATA UPLOAD PORT: CSV HEADER MAPPER =================
SAMPLE_CSV_HEADER = ['district', 'date', 'epi_week', 'reporting_year', 'suspected_cases', 'rdt_confirmations']
SAMPLE_CSV_ROWS = [
    ['Chadiza', '2026-01-05', '1', '2026', '120', '68'],
    ['Chadiza', '2026-01-12', '2', '2026', '145', '81'],
    ['Lusaka', '2026-01-05', '1', '2026', '310', '150'],
    ['Lusaka', '2026-01-12', '2', '2026', '298', '142'],
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
            susp_idx = find_index(['susp', 'attend', 'total'], 4)
            rdt_idx  = find_index(['rdt', 'confirm', 'pos'], 5)
            pop_idx  = find_optional_index(['pop'])  # optional — enables incidence-rate hotspot tiers

            def safe_int(value_str):
                try:
                    return int(float(str(value_str).strip()))
                except (ValueError, TypeError):
                    return 0

            created_count = 0
            updated_count = 0
            for row in reader:
                if not row or len(row) <= max(dist_idx, date_idx, week_idx, year_idx, susp_idx, rdt_idx):
                    continue

                district_name = row[dist_idx].strip()
                record_date_str = row[date_idx].strip()
                epi_week      = safe_int(row[week_idx])
                reporting_year = safe_int(row[year_idx])
                suspected     = safe_int(row[susp_idx])
                rdt_positives = safe_int(row[rdt_idx])

                if not district_name:
                    continue

                district_obj, _district_created = ZambianDistrict.objects.get_or_create(
                    name=district_name,
                    defaults={'province': 'Surveillance Region', 'latitude': -15.42, 'longitude': 28.28}
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
                        'suspected_cases': suspected,
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

        except Exception as error:
            messages.error(request, f"Failed to parse spreadsheet file structure. Technical log error: {str(error)}")

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
            target_user = SystemUser.objects.filter(pk=target_id).first()
            if not target_user:
                messages.error(request, "Account not found.")
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
