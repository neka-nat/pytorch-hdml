import argparse
import visdom

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send visdom window data.')
    parser.add_argument('input_log', action='store', type=str, help="Input log file.")
    parser.add_argument('-v', '--visdomserver', type=str, default='localhost', help="Visdom's server name.")
    args = parser.parse_args()
    viz = visdom.Visdom(server='http://' + args.visdomserver)
    assert viz.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
    viz.replay_log(args.input_log)