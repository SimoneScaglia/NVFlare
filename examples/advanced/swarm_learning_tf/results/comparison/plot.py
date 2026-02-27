import pandas as pd
import matplotlib.pyplot as plt

# Configurazione del grafico
plt.figure(figsize=(10, 6))
plt.xlabel('Splits')
plt.ylabel('AUC')
plt.title('AUC vs Splits Comparison')
plt.grid(True)

# Definizione dei colori per i diversi modelli
colors = {
    'central': 'blue',
    'swarm': 'green',
    'federated': 'red',
    'personalized_swarm': 'purple',
    'personalized_federated': 'orange'
}

# Definizione degli splits per swarm e federated
swarm_splits_mapping = {
    24.375: 2.5,
    23.75: 5,
    22.5: 10,
    20: 20,
    17.5: 30,
    15: 40,
    12.5: 50,
    10: 60,
    7.5: 70,
    5: 80,
    2.5: 90
}

swarm_data = pd.read_csv('swarm_aucs.csv')
swarm_means = []
for original_split, target_split in swarm_splits_mapping.items():
    split_data = swarm_data[swarm_data['splits'] == original_split]
    mean_auc = split_data['auc'].mean()
    swarm_means.append((target_split, mean_auc))
swarm_means.sort(key=lambda x: x[0])
swarm_splits, swarm_aucs = zip(*swarm_means)
plt.plot(swarm_splits, swarm_aucs, 'o-', color=colors['swarm'], 
        label='Swarm')

federated_data = pd.read_csv('federated_aucs.csv')
federated_means = []
for original_split, target_split in swarm_splits_mapping.items():
    split_data = federated_data[federated_data['splits'] == original_split]
    mean_auc = split_data['auc'].mean()
    federated_means.append((target_split, mean_auc))
federated_means.sort(key=lambda x: x[0])
federated_splits, federated_aucs = zip(*federated_means)
plt.plot(federated_splits, federated_aucs, 'o-', color=colors['federated'], 
        label='Federated')

central_data = pd.read_csv('central_aucs.csv')
central_mean_auc = central_data['auc'].mean()
plt.hlines(central_mean_auc, min(swarm_splits), max(swarm_splits), 
        colors=colors['central'], linestyles='dashed', 
        label=f'Centralized')

# Aggiunta legenda e visualizzazione grafico
plt.legend()
plt.xticks(list(swarm_splits_mapping.values()))
plt.tight_layout()
plt.savefig('plot.png')
