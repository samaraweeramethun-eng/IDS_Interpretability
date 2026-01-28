from methun_research.config import EnhancedConfig
from methun_research.training import train_enhanced


def main():
    config = EnhancedConfig()
    train_enhanced(config)


if __name__ == "__main__":
    main()
