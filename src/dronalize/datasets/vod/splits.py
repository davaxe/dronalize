from dronalize.core.categories import DatasetSplit

train = [
    "q4hez5edjjtfph90o7y22t1tdgnnqfkp",
    "8k2fk99l0phbz83lxtc25q87e06jarnk",
    "owogq5dslsmf48dagon6fxeb108ts8eh",
    "8ljts8qndgju0bcdh3ldez8y34jl50u8",
    "xfoiw0w00ggr42lp2mwt7sanxjm9wtuy",
    "okwsl2921sjnw8t017nb0aq6y9k29w8r",
    "0mlrodqf39vvdidiifyvlw313rnahyht",
    "mkr529kg3p45sk0jgrjd8szfegwfot79",
    "mypp0mldnv4kenlr3gqt8638qv9f7pkk",
    "7kjhncjj4m5883qwu0rjo1j62hu8hdx8",
    "f7b4df1owz80izwy51roqruva31ssmqn",
    "87cgygi3l41q3ft0ct9dizxclqg4b434",
]

train_val = [
    "smsz20lamf98gmpt1ppia0oefubih4kw",
    "63p8qt0d4biowur5es45sojf7fj3h5iu",
    "9q5jy0dlaz7blaw6fd1fizdnnw2dl4to",
    "0j93nt1dpgzxvhppg9smwlul7ka75rb4",
]

test = [
    "r4bf59dp37h696i9gg03siw2gq8p4msn",
    "k3rblq6f3tulyggqfyjz8st4ewokv8fa",
    "79fxw6jjbwgw8lax0ukcwtrujv4ktj5v",
    "4tym7cobttc56enha5uepsyi3xhtmpgy",
    "bnqhhjrb8zn6frz29emtekjk7qwo3q3p",
    "2vrb1jiswk9ocphpj1fzouby4d2f9cow",
    "wvbc8v977mciwp2b7qyxerv0xyw6sr60",
]


def get_split(token: str) -> DatasetSplit:
    """Determine the split based on token."""
    if token in train:
        return DatasetSplit.TRAIN
    if token in train_val:
        return DatasetSplit.VAL
    if token in test:
        return DatasetSplit.TEST
    msg = f"Token {token} does not belong to any known split."
    raise ValueError(msg)
