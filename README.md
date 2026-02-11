# Local Watt v1

Offline ERG-only trainer app focused on reliable KICKR CORE BLE FTMS control.

## Scope

- Single rider profile.
- One trainer connection (KICKR CORE) over BLE FTMS.
- Optional Bluetooth heart-rate monitor connection (standard Heart Rate service).
- Structured workout execution in ERG mode only.
- Workout import: JSON and optional ZWO.
- Controls: Start, Pause, Resume, Skip, Back, End.
- Intensity bias: 90% to 110%.
- Local session storage (SQLite).
- Post-ride summary and export (CSV and FIT when `fit-tool` is available).

Out of scope in v1:

- SIM/resistance modes.
- Route/world gameplay.
- Multiplayer/ghosts.

## Run from source

```powershell
python app.py
```

## Build standalone exe

```powershell
.\build.ps1 -Clean
```

Output:

- `dist\LocalWatt.exe`

## Workout JSON format

```json
{
  "id": "tempo_builder_28",
  "name": "Tempo Builder 28m",
  "ftp": 250,
  "steps": [
    { "type": "warmup", "duration_s": 300, "target_pct_ftp": 0.55 },
    { "type": "steady", "duration_s": 480, "target_pct_ftp": 0.75 }
  ]
}
```

Runtime target:

- `target_watts = round(ftp * target_pct_ftp * intensity_bias)`
- Clamped to `80W` minimum and `150% FTP` ceiling.

## Data files

- SQLite DB: `%LOCALAPPDATA%\LocalWatt\localwatt.db`
- Exports: `%LOCALAPPDATA%\LocalWatt\exports`
