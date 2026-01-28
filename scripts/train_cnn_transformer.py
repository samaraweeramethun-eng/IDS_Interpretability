from methun_research.config import CNNTransformerConfig
from methun_research.training import train_cnn_transformer


def main():
    config = CNNTransformerConfig()
    train_cnn_transformer(config)


if __name__ == "__main__":
    main()
