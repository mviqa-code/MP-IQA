import argparse

from scipy import stats


def main(args):

    quality_scores = []
    mAPs = []

    with open(args.iqa_file, 'r', encoding='utf-8') as file:
        _file = sorted(file)
        for _line in _file:
            score = float(_line.split('\t')[1].strip())
            quality_scores.append(score)

    with open(args.map_file, 'r', encoding='utf-8') as file:
        file = sorted(file)
        for line in file:
            map = float(line.split('\t')[1].strip())
            mAPs.append(map)

    plcc, p1 = stats.pearsonr(quality_scores, mAPs)
    srcc, p2 = stats.spearmanr(quality_scores, mAPs)
    krcc, p3 = stats.kendalltau(quality_scores, mAPs)
    print(f'PLCC:{plcc}, SRCC:{srcc}, KRCC:{krcc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--iqa_file',
        default='/DATA2/chenyuqing_program/MP-IQA-main/results/iqa_score/trashcan/mpiqa/wolocation.txt'
    )
    parser.add_argument(
        '--map_file',
        default='/DATA2/chenyuqing_program/MP-IQA-main/results/image_map/trashcan/faster-rcnn/map.txt'
    )
    main(parser.parse_args())