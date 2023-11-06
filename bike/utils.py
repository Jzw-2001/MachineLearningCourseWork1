SEASON2NUM = {
    'Spring': 0,
    'Summer': 1,
    'Autumn': 2,
    'Winter': 3
}

HOLIDAY2NUM = {
    'No Holiday': 0,
    'Holiday': 1
}

FUNCTIONINGDAY2NUM = {
    'No': 0,
    'Yes': 1
}

def debug_log(msg, file):
    with open(file, 'a+') as f:
        f.write(msg + '\n')