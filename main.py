import click

@click.command()
@click.argument("option")
@click.option("--option", "-o", default='train', help="Режим работы, train или eval")
@click.help_option('--help','-h',help='Показывает инструкции')
def hello(option):
    """Меню позволяющее выбрать режим программы"""
    match option:
        case 'train':
            print(20*'_')
            print('ЗАПУСКА ОБУЧЕНИЯ')
            print(20*'_')
            import train
        case 'eval':
            print(20*'_')
            print('ЗАПУСК ТЕСТИРОВАНИЯ')
            print(20*'_')
            import eval
    click.echo(f"Выбран {option} режим!")

if __name__ == "__main__":
    hello()
