#!/bin/bash
# declare -a xarr=("vodca")

# for x in "${xarr[@]}"
# do
# 	if [ $x == "vodca" ]
# 	then
# 		python 2-TrainModel.py --X $x --Y inat_gbif --filter-outliers --resume
# 	fi
# done

# python -m 2-TrainModel --pft Shrub_Tree --X-collection "data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat=any_thr=0.5.parquet"
# python -m 2-TrainModel --pft Grass --X-collection "data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat=any_thr=0.5.parquet"
# python -m scripts.predict_maps --new data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_nan-strat\=any_thr\=0.5 --tiled --run-ids "2023-09-23_13-30-09" "2023-09-23_11-44-08"

# python -m scripts.predict_maps --new data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_2_deg_nan-strat\=any_thr\=0.5.parquet --new-imputed data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_2_deg_nan-strat\=any_thr\=0.5_imputed.parquet -p Grass --aoa
# python -m scripts.predict_maps --new data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat\=any_thr\=0.5.parquet --new-imputed data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat\=any_thr\=0.5_imputed.parquet -p Shrub_Tree
# python -m scripts.predict_maps --new data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat\=any_thr\=0.5.parquet --new-imputed data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat\=any_thr\=0.5_imputed.parquet -p Grass
# python -m scripts.predict_maps --new data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.2_deg_nan-strat\=any_thr\=0.5.parquet --new-imputed data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.2_deg_nan-strat\=any_thr\=0.5_imputed.parquet -p Shrub_Tree
# python -m scripts.predict_maps --new data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.2_deg_nan-strat\=any_thr\=0.5.parquet --new-imputed data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.2_deg_nan-strat\=any_thr\=0.5_imputed.parquet -p Grass
python -m scripts.predict_maps --new data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_nan-strat\=any_thr\=0.5 --tiled -p Shrub_Tree --num-procs 10
python -m scripts.predict_maps --new data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_nan-strat\=any_thr\=0.5 --tiled -p Grass --num-procs 10
python -m scripts.predict_maps --new data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_nan-strat\=any_thr\=0.5 --tiled --run-ids "2023-09-23_13-30-09" "2023-09-23_11-44-08" --num-procs 5