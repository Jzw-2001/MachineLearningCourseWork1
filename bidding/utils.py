def debug_log(msg, file):
    with open(file, 'a+') as f:
        f.write(msg + '\n')