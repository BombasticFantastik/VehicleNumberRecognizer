import click

@click.command()
@click.argument("option")
@click.option("--option", "-o", default='train', help="Режим работы, train или eval")
def hello(option):
    """Меню позволяющее выбрать режим программы, eval """
    match option:
        case 'train':
            pass
        case 'eval':
            pass
    click.echo(f"Выбран {option} режим!")

if __name__ == "__main__":
    hello()
