from environment.sumo_env import SumoEnvironment

def main():

    env = SumoEnvironment()

    env.show_configuration()

    env.connect()

    env.disconnect()


if __name__ == "__main__":
    main()