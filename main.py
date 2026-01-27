import click
import os
import yaml


option_path='config.yaml'
with open(option_path,'r') as file_option:
    files_option=yaml.safe_load(file_option)

@click.command()
@click.argument("option", default='train')
@click.argument("count", default=1)
@click.help_option('--help','-h',help='Показывает инструкции')


    


def hello(option,count):
    """Меню позволяющее выбрать режим программы"""
    match option:
        case 'train':
            from train import Train_from_main
            print(20*'_')
            print('ЗАПУСКА ОБУЧЕНИЯ')
            print(20*'_')
            Train_from_main(count)
                
        case 'eval':
            from eval import Eval_from_main
            print(20*'_')
            print('ЗАПУСК ТЕСТИРОВАНИЯ')
            print(20*'_')
            Eval_from_main(count)
        case 'drop':
            print(20*'_')
            print('СБРОС ВЕСОВ')
            print(20*'_')
            os.remove(files_option['weights'])
    #click.echo(f"Выбран {option} режим!")

if __name__ == "__main__":
    hello()
