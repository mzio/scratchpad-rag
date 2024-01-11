import argparse


def main():
    parser = argparse.ArgumentParser(description='Make seeds')
    parser.add_argument('--script', type=str, default='')
    parser.add_argument('--num_seeds', type=int, default=5)
    
    args = parser.parse_args()

    all_scripts = args.script
    print(f'\nScripts:')
    print(f'--------')
    for script in all_scripts.split('\n'):
        if '--seed' in script:
            seed = int(script.split('--seed ')[-1].split(' --')[0])
            for _seed in range(args.num_seeds):
                print(script.replace(f'--seed {seed}', f'--seed {_seed}'))
                # print('\n')
            print('\n')
        elif script == '':
            pass
        else:
            raise Exception('Please specify --seed argument in script')
        
    
if __name__ == '__main__':
    main()