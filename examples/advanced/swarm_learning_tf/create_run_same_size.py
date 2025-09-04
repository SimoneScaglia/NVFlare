DEFAULT_ITER = 15

with open('run_same_size.sh', 'w') as f:
    for nodes in range(2, 26):
        for iter in range(DEFAULT_ITER):
            value = round(100 / nodes, 2)
            params = " ".join([str(value)] * nodes)
            command = f"./start_exp_same_size.sh {iter} {nodes} {params}\n"
            f.write(command)
