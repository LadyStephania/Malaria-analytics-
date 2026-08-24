import os
import sys
import json
import time
import statistics
import requests

sys.path.insert(0, r'C:\Users\stephania.bwalya\Desktop\Malaria\core_project')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core_project.settings')
import django
django.setup()

from malaria.models import ZambianDistrict  # noqa: E402
import csv

NAME_FIX = {
    'Itezhi-Tezhi': 'Itezhi-tezhi',
    'Mushindamo': 'Mushindano',
    "Shang'ombo": 'Shangombo',
}

with open(r'C:\Users\stephania.bwalya\Desktop\zambia_monthly_malaria_simulated.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    csv_districts = set(row['District'].strip() for row in reader)
csv_districts = {NAME_FIX.get(d, d) for d in csv_districts}

districts = list(ZambianDistrict.objects.filter(name__in=csv_districts).values('name', 'latitude', 'longitude'))
print(f'Fetching weather for {len(districts)} districts (expected {len(csv_districts)})')
missing = csv_districts - {d['name'] for d in districts}
if missing:
    print('WARNING - not found in DB:', missing)

OUT = r'C:\Users\STEPHA~1.BWA\AppData\Local\Temp\claude\c--Users-stephania-bwalya-Desktop-Malaria\5aee0dd0-b7de-4b7f-bd0f-05cb0e61b724\scratchpad\weather_monthly.json'

results = {}
if os.path.exists(OUT):
    with open(OUT, encoding='utf-8') as f:
        results = json.load(f)

for i, d in enumerate(districts, 1):
    name = d['name']
    if name in results:
        continue
    url = (
        'https://archive-api.open-meteo.com/v1/archive'
        f"?latitude={d['latitude']}&longitude={d['longitude']}"
        '&start_date=2023-01-01&end_date=2025-12-31'
        '&daily=precipitation_sum,temperature_2m_mean&timezone=Africa%2FLusaka'
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()['daily']
            break
        except Exception as e:
            print(f'  retry {name}: {e}')
            time.sleep(2)
    else:
        print(f'FAILED: {name}')
        continue

    monthly = {}
    for date_str, precip, temp in zip(data['time'], data['precipitation_sum'], data['temperature_2m_mean']):
        ym = date_str[:7]  # YYYY-MM
        monthly.setdefault(ym, {'precip': [], 'temp': []})
        if precip is not None:
            monthly[ym]['precip'].append(precip)
        if temp is not None:
            monthly[ym]['temp'].append(temp)

    agg = {}
    for ym, vals in monthly.items():
        agg[ym] = {
            'rainfall_mm': round(sum(vals['precip']), 1) if vals['precip'] else None,
            'avg_temperature_c': round(statistics.mean(vals['temp']), 1) if vals['temp'] else None,
        }
    results[name] = agg
    print(f'[{i}/{len(districts)}] {name}: {len(agg)} months')

    if i % 10 == 0:
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(results, f)

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f)
print('DONE. Saved to', OUT)
