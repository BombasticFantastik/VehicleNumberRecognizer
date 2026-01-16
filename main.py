import click
import os
from train import Train_from_main
from eval import Eval_from_main

@click.command()
@click.argument("option", default='train')
@click.argument("count", default=1)
@click.help_option('--help','-h',help='Показывает инструкции')
def hello(option,count):
    """Меню позволяющее выбрать режим программы"""
    match option:
        case 'train':
            print(20*'_')
            print('ЗАПУСКА ОБУЧЕНИЯ')
            print(20*'_')
            Train_from_main(count)
                
        case 'eval':
            print(20*'_')
            print('ЗАПУСК ТЕСТИРОВАНИЯ')
            print(20*'_')
            Eval_from_main(count)
        case 'drop':
            print(20*'_')
            print('СБРОС ВЕСОВ')
            print(20*'_')
            os.remove('/home/artemybombastic/MyGit/VehicleNumberData/VNR_Data/weights/crnn_weights.pth')
    #click.echo(f"Выбран {option} режим!")

if __name__ == "__main__":
    hello()
