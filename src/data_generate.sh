cd ./dataprocess
python preprocess_hdfs.py
python preprocess_hdfs.py --anomaly_ratio 0.0
python preprocess_non_hdfs.py --DS bgl
python preprocess_non_hdfs.py --DS bgl --anomaly_ratio 0.0
python preprocess_non_hdfs.py --DS spirit
# python preprocess_non_hdfs.py --DS spirit --anomaly_ratio 0.0
python preprocess_non_hdfs.py --DS tbd
python preprocess_non_hdfs.py --DS tbd --anomaly_ratio 0.0

python graph_preprocess.py
python graph_preprocess.py --anomaly_ratio 0.0
cd ..