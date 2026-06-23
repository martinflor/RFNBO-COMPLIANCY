import subprocess
import csv
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]
PY = WORKDIR / '.venv' / 'Scripts' / 'python.exe'
CLI = WORKDIR / 'rfnbo_capacity_factor_sweep_cli.py'
GEN = WORKDIR / 'entsoe_data' / 'Belgium' / 'Belgium_generation_20200101_20251220.csv'
INST = WORKDIR / 'entsoe_data' / 'Belgium' / 'Belgium_installed_capacity_20200101_20251220.csv'
OUTDIR = WORKDIR / 'outputs'
OUTDIR.mkdir(exist_ok=True)

summary_file = OUTDIR / 'Belgium_capacity_sweep_summary.csv'

ratios = [round(i * 0.1, 2) for i in range(1, 21)]

with open(summary_file, 'w', newline='') as sf:
    writer = csv.writer(sf)
    writer.writerow(['ratio', 'tech', 'capacity_mw', 'annual_energy_mwh', 'cf'])

    for r in ratios:
        out_csv = OUTDIR / f'Belgium_ppa_series_ratio_{r:.2f}.csv'
        cmd = [str(PY), str(CLI),
               '--electrolyser-mw', '50',
               '--base-ppa-ratio', str(r),
               '--tech-mix', 'offshore:1',
               '--use-data',
               '--generation-csv', str(GEN),
               '--installed-csv', str(INST),
               '--temporal',
               '--out-csv', str(out_csv)
               ]
        print(f'Running ratio {r}...')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout
        # parse sizing lines
        for line in stdout.splitlines():
            if line.strip().startswith('- '):
                # format: - offshore: capacity 166.667 MW -> annual energy 438000 MWh (cf=0.300)
                try:
                    part = line.strip()[2:]
                    tech, rest = part.split(':', 1)
                    tech = tech.strip()
                    cap_part, rest2 = rest.split('->', 1)
                    cap_val = cap_part.replace('capacity', '').replace('MW', '').strip()
                    energy_part, cf_part = rest2.split('(', 1)
                    energy_val = energy_part.replace('annual energy', '').replace('MWh', '').strip()
                    cf_val = cf_part.replace('cf=', '').replace(')', '').strip()
                    writer.writerow([r, tech, float(cap_val), float(energy_val), float(cf_val)])
                except Exception as e:
                    print('Failed to parse line:', line, e)

print('Sweep finished. Summary saved to', summary_file)
