# HebSeam Reusable Release

The release contains two independent parts.

## 1. `01_hebseam_database`

Public, reusable benchmark generator. It downloads source corpora, generates a seeded database, records provenance and ground truth, and validates the resulting database.

## 2. `02_hebseam_methods`

Independent experimental code. It accepts the path of any compatible HebSeam database. The database can remain outside this repository and can be uploaded, downloaded, generated, or versioned separately.

The methods never import Python modules from `01_hebseam_database`; the only interface between the two parts is the generated database folder.

Typical workflow:

```bash
cd 01_hebseam_database
python -m hebseam_benchmark.fetch_data --data-dir data/raw
python build_benchmark.py --config config.yaml --data-dir data/raw --output-dir data/generated/hebseam-v1.0.0 --force
python validate_benchmark.py data/generated/hebseam-v1.0.0

cd ../02_hebseam_methods
pip install -r requirements.txt
python run_all.py ../01_hebseam_database/data/generated/hebseam-v1.0.0 --results results
```
