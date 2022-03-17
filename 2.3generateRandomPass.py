from passwd.passTool import generate_random_pass, three_sigma_deduce

if __name__ == '__main__':
    generate_random_pass(300000)
    three_sigma_deduce("raw_dataset/random_pass.csv")