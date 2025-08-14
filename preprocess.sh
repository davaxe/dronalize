python -m preprocessing.preprocess_urban --config 'rounD' --debug $1 --use-threads $2 --add-supp 0
python -m preprocessing.preprocess_urban --config 'inD' --debug $1 --use-threads $2 --add-supp 0
python -m preprocessing.preprocess_urban --config 'uniD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_urban --config 'SIND' --debug $1 --use-threads $2
python -m preprocessing.preprocess_highway --config 'us101' --debug $1 --use-threads $2 --add-supp 0
python -m preprocessing.preprocess_highway --config 'i80' --debug $1 --use-threads $2 --add-supp 0
python -m preprocessing.preprocess_highway --config 'highD' --debug $1 --use-threads $2 --add-supp 0
python -m preprocessing.preprocess_highway --config 'exiD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_highway --config 'A43' --debug $1 --use-threads $2
python -m preprocessing.preprocess_highway --config 'ad4che' --debug $1 --use-threads $2
python -m preprocessing.preprocess_interact --config 'interact' --debug $1 --use-threads $2
python -m preprocessing.preprocess_apollo --config 'apollo' --debug $1 --use-threads $2
python -m preprocessing.preprocess_argoverse --config 'argoverse' --debug $1 --use-threads $2
python -m preprocessing.preprocess_av2 --config 'av2' --debug $1 --use-threads $2
python -m preprocessing.preprocess_nuscenes --config 'nuscenes' --debug $1 --use-threads $2
python -m preprocessing.preprocess_waymo --config 'waymo' --debug $1 --use-threads $2
python -m preprocessing.preprocess_lyft --config 'lyft' --debug $1 --use-threads $2
python -m preprocessing.preprocess_opendd --config 'opendd' --debug $1 --use-threads $2
python -m preprocessing.preprocess_vod --config 'vod' --debug $1 --use-threads $2


## Apptainer
# apptainer run /path/to/dronalize.sif python -m preprocessing.preprocess_urban --config 'rounD' --debug $1 --use-threads $2 --add-supp 0

## Docker
# docker run -v "$(pwd)":/app -v "$(pwd)/../datasets":/datasets -w /app dronalize-dev python -m preprocessing.preprocess_urban --config 'rounD' --debug $1 --use-threads $2 --add-supp 0

