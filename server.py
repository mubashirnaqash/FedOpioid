
import flwr as fl
import csv
import matplotlib.pyplot as plt
import pandas as pd

class CustomServerStrategy(fl.server.strategy.FedAvg):
    def __init__(self, fraction_fit, fraction_eval, min_fit_clients, min_eval_clients, min_available_clients):
        super().__init__(fraction_fit, fraction_eval, min_fit_clients, min_eval_clients, min_available_clients)
        self.eval_metrics = []
        with open('federated_metrics.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["round", "accuracy", "roc_auc", "precision", "recall", "f1_score"])

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_weights = super().aggregate_evaluate(rnd, results, failures)
        
        # Collect additional metrics
        additional_metrics = {
            "accuracy": [],
            "roc_auc": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
        }
        for _, result in results:
            metrics = result.metrics
            for key in additional_metrics.keys():
                if key in metrics:
                    additional_metrics[key].append(metrics[key])
        
        # Calculate average of each metric
        avg_metrics = {key: sum(vals) / len(vals) for key, vals in additional_metrics.items()}
        self.eval_metrics.append(avg_metrics)
        print(f"Round {rnd}: Average metrics: {avg_metrics}")

        # Write average metrics to CSV file
        with open('federated_metrics.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([rnd] + list(avg_metrics.values()))

        return aggregated_weights

def create_strategy():
    return CustomServerStrategy(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3
    )

def plot_aggregated_metrics(csv_path):
    metrics_df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    for metric in metrics_df.columns[1:]:  # Skip the round column
        plt.plot(metrics_df['round'], metrics_df[metric], label=metric)

    plt.title('Aggregated Metrics Over Rounds')
    plt.ylabel('Value')
    plt.xlabel('Round')
    plt.legend()
    plt.show()

def start_server():
    strategy = create_strategy()
    fl.server.start_server("localhost:8080", config={"num_rounds": 3}, strategy=strategy)
    plot_aggregated_metrics('federated_metrics.csv')

if __name__ == "__main__":
    start_server()
