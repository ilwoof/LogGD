from argparse import ArgumentParser
import os
from data_process import process_dataset
import sys
import time

sys.path.append("../../")

log_file_name = {
               'hdfs': 'HDFS.log',
               'bgl': 'BGL.log',
               'tbd': 'Thunderbird10M.log',
               'spirit': 'Spirit1G.log',
               }


def arg_parser():
    """
    add parser parameters
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=['hdfs', 'bgl', 'spirit', 'tbd'], help="dataset to use")
    parser.add_argument("--data_dir", default="../../dataset/raw/", metavar="DIR", help=" input data directory")
    parser.add_argument("--output_dir", default="../../dataset/processed/", metavar="DIR", help="output directory")

    parser.add_argument("--window_type", default="session", type=str, choices=["sliding", "session"])
    parser.add_argument("--session_level", default="entry", type=str, choices=["entry", "mins"])
    parser.add_argument('--window_size', nargs="+", default=[100, 20], type=float, help='window size/mins') # only entry is considered
    # parser.add_argument('--step_size', default=1, type=float, help='step size/mins')  # for our case, only fixed_window is considered
    parser.add_argument('--train_size', default=0.8, type=float, help="train size", metavar="float or int")

    # features
    parser.add_argument("--is_logkey", default=True, help="Whether logkey included in features")
    parser.add_argument("--random_sample", action='store_true', help="Whether random sampling are conducted")
    parser.add_argument("--is_time", action='store_true', help="is time duration included in features")

    return parser


if __name__ == '__main__':
    print(os.getcwd())
    parser = arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    for dataset in args.datasets:
        data_dir = os.path.expanduser(f'{args.data_dir}/{dataset}/')
        # data_dir = os.path.expanduser(f'{args.data_dir}')
        if dataset == 'hdfs':
            window_type = 'session'
            windows = ['session']
        else:
            window_type = 'sliding'
            windows = args.window_size

        for window_size in windows:

            # Since we only consider fixed windows, step_size is always set equal to the window size
            step_size = window_size

            output_dir = os.path.join(args.output_dir, f"{dataset}")

            process_dataset(data_dir=data_dir, output_dir=output_dir,
                            log_file=log_file_name[dataset],
                            dataset_name=dataset, window_type=window_type,
                            window_size=window_size, step_size=step_size,
                            train_size=args.train_size, random_sample=args.random_sample,
                            session_type=args.session_level)
