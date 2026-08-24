import csv
import datetime

SRC = r'C:\Users\stephania.bwalya\Desktop\zambia_monthly_malaria_simulated.csv'
DST = r'C:\Users\STEPHA~1.BWA\AppData\Local\Temp\claude\c--Users-stephania-bwalya-Desktop-Malaria\5aee0dd0-b7de-4b7f-bd0f-05cb0e61b724\scratchpad\simulated_for_upload.csv'

# Normalize the 3 district names that differ from the DB's existing (real,
# already-geocoded/census) spelling, so they attach to the existing district
# row instead of spawning a duplicate with dummy Lusaka fallback coordinates.
NAME_FIX = {
    'Itezhi-Tezhi': 'Itezhi-tezhi',
    'Mushindamo': 'Mushindano',
    "Shang'ombo": 'Shangombo',
}

with open(SRC, encoding='utf-8') as f_in, open(DST, 'w', newline='', encoding='utf-8') as f_out:
    reader = csv.DictReader(f_in)
    writer = csv.writer(f_out)
    writer.writerow([
        'district', 'date', 'epi_week', 'reporting_year', 'rdt_confirmations',
        'suspected_cases', 'rdt_tested', 'microscopy_tested',
    ])
    n = 0
    for row in reader:
        district = row['District'].strip()
        district = NAME_FIX.get(district, district)
        year = int(row['Year'])
        month = int(row['Month'])
        date = datetime.date(year, month, 1)
        epi_week = date.isocalendar()[1]
        confirmed = row['Confirmed_Cases'].strip()
        suspected = row['Suspected_Cases'].strip()
        rdt_tested = row['RDT_Tested'].strip()
        microscopy_tested = row['Microscopy_Tested'].strip()
        writer.writerow([
            district, date.isoformat(), epi_week, year, confirmed,
            suspected, rdt_tested, microscopy_tested,
        ])
        n += 1

print(f'Wrote {n} rows to {DST}')
