from ExampleAgent import ExampleAgent

if __name__ == "__main__":
    agent = ExampleAgent(config_file='config.json')
    avg = agent.rec_post({'readings': [60, 65, 70, 75, 80]})
    print(avg)