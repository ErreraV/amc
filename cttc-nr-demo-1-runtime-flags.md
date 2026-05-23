# cttc-nr-demo-1 — Runtime flags & parameters

This file documents the CLI flags and common runtime parameters you can set when running the `cttc-nr-demo-1` simulation (source: `src/amc_/demo/cttc-nr-demo-1.cc`). Flag names are case-sensitive.

## Supported CLI flags

| Flag | Type | Default | Description | Units / Notes |
|---|---:|---:|---|---|
| `--gNbNum` | integer | `1` | Number of gNBs | |
| `--ueNumPergNb` | integer | `10` | Number of UEs per gNB | |
| `--logging` | bool | `false` | Enable info logging for components | pass `true` or `1` to enable |
| `--simTime` | Time | `60s` | Total simulation time | Accepts units: `s`, `ms`, `us`, `ns` (case-sensitive flag name)
| `--maxSpeed` | double | `80.0` | Maximum UE speed | km/h |
| `--minSpeed` | double | `5.0` | Minimum UE speed | km/h |
| `--areaWidth` | double | `2000.0` | Simulation area width | meters |
| `--areaHeight` | double | `2000.0` | Simulation area height | meters |
| `--channelType` | string | `urban` | Channel type: `urban`, `suburban`, `rural` | affects channel model selection |
| `--ganDataFile` | string | `gan_data_.json` | Name of the exported GAN training data file | relative to `--outputDir` |
| `--outputDir` | string | `./` | Output directory for stats and GAN file | |
| `--mobilityType` | integer | `5` | Mobility type (0=STATIC,1=LINEAR,2=CIRCULAR,3=RANDOM_WALK,4=HIGHWAY,5=URBAN_GRID) | integer code |
| `--packetSize` | integer | `64` | Packet size used by internal logic (`global_packet_size`) | bits (maps to global packet size used by Sionna/feedback calls)


## Other important runtime parameters (set in code)

- `udpAppStartTime` — default `5s`: start time for UDP client/server applications. Not exposed as a CLI flag in the file; change in source if needed.
- `centralFrequencyBand1` — default `28e9` (Hz): carrier frequency used to initialize channel config.
- `totalTxPower` — default `35` (dB scale used to compute Tx power in code).
- `data_collection_interval` — default `0.1` (s): interval used to schedule `ScheduleDynamicTransmissions` events.

To permanently change these, edit their initializers in `main()` and rebuild.

## Tips & examples

- Use the exact flag capitalization — `--simTime` (capital `T`) is required by the `CommandLine` parser in the file. Passing `--simtime` will be reported as an invalid option.
- Example runs with waf:

```sh
./waf --run "scratch/cttc-nr-demo-1 --simTime=120s --maxSpeed=60 --areaWidth=1500 --channelType=suburban --packetSize=1024"
```

- Example running built binary directly (if you built outside waf):

```sh
./cttc-nr-demo-1 --simTime=300s --mobilityType=4 --minSpeed=2 --packetSize=512
```

- If you want a lowercase alias for convenience, add in source near the other `AddValue` calls:

```cpp
cmd.AddValue("simtime", "alias for simTime (lowercase)", simTimeNs3);
```

After editing, rebuild the program.

## Where to edit

- File: `src/amc_/demo/cttc-nr-demo-1.cc` — look in `main()` for the `CommandLine` block and the initial variable values.

If you want, I can patch the source to add a lowercase alias (`--simtime`) or expose additional flags (for example `--udpAppStartTime`). Tell me which change you prefer and I'll apply it.
