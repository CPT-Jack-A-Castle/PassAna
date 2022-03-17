from passwd.passTool import three_sigma_deduce, generate_random_token

if __name__ == '__main__':
    generate_random_token(300000)
    three_sigma_deduce("raw_dataset/tokens.csv")