python -m scripts.save_collection --res-str "0.2_deg" --nan-strategy "any" --thresh 0.5
python -m scripts.save_collection --res-str "2_deg" --nan-strategy "any" --thresh 0.5
python -m scripts.save_collection --collection "data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.2_deg_nan-strat=any_thr=0.5.parquet" --impute-missing
python -m scripts.save_collection --collection "data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_2_deg_nan-strat=any_thr=0.5.parquet" --impute-missing