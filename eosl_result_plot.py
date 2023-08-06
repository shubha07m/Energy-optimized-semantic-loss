from modified_eosl import eosl_loss
import pandas as pd
import matplotlib.pyplot as plt


def get_eosl_stats(plot_graph=0):
    enc_list = ['vit', 'gitbase', 'blipbase', 'gitlarge', 'bliplarge']
    k = 1
    if plot_graph:
        i = 0
        dfs = {}

        for enc in enc_list:
            df = pd.read_csv('eosl_plotdata.csv')
            df_data = df[df['encoder'] == enc]
            dfs[i] = df_data
            i += 1
        # Plot dataset 1 as a line plot
        plt.plot(dfs[0]['pb'], dfs[0]['eosl'], label='vit', marker='x')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[1]['pb'], dfs[1]['eosl'], label='gitbase', marker='+')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[2]['pb'], dfs[2]['eosl'], label='blipbase', marker='*')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[3]['pb'], dfs[3]['eosl'], label='gitlarge', marker='o')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[4]['pb'], dfs[4]['eosl'], label='bliplarge', marker='p')

        # Add labels, title, and legend
        legend_font = {'weight': 'bold', 'size': 10}
        plt.xlabel('Probability of bit error', fontweight='bold', size=12)
        plt.ylabel('Energy Optimized Sematic Loss', fontweight='bold', size=12)
        plt.suptitle('EOSL v bit error rate for various encoder', fontweight='bold', size=12)
        plt.title('K= ' + str(k), fontweight='bold', size=12)
        plt.legend(loc='lower right', frameon=False, prop=legend_font)

        # Show the plot
        plt.show()

    else:
        plot_data_all = []
        pb_list = [.001, .01, .05, .1]
        plot_data_all.append(['encoder', 'pb', 'eosl'])
        for enc in enc_list:
            for pb in pb_list:
                plot_data_all.append([enc, pb, eosl_loss(enc, pb)])

        plot_data_panda = pd.DataFrame(plot_data_all)
        plot_data_panda.to_csv('eosl_plotdata.csv', index=False, header=False)


if __name__ == '__main__':
    print('add new plotdata or create graph: 1 for graph, 0 for else')
    get_eosl_stats(int(input()))
