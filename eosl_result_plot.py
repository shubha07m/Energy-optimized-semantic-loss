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
            df_part = df[df['pf'] == .005]
            df_plotdata = df_part[df_part['encoder'] == enc]
            dfs[i] = df_plotdata
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
        plt.xlabel('Probability of bit error')
        plt.ylabel('Energy Optimized Sematic Loss')
        plt.suptitle('EOSL v bit error rate for various encoder')
        plt.title('K= ' + str(k))
        plt.legend()

        # Show the plot
        plt.show()

    else:
        plot_data_all = []
        pb_list = [.1, .3, .5, .7, .9]
        pf_list = [.001, .003, .005, .007, .009]
        plot_data_all.append(['encoder', 'pb', 'pf', 'eosl'])
        for enc in enc_list:
            for pb in pb_list:
                for pf in pf_list:
                    plot_data_all.append([enc, pb, pf, eosl_loss(pb, pf, k, 1, enc)])

        plot_data_panda = pd.DataFrame(plot_data_all)
        plot_data_panda.to_csv('eosl_plotdata.csv', index=False, header=False)


if __name__ == '__main__':
    print('add new plotdata or create graph: 1 for graph, 0 for else')
    get_eosl_stats(int(input()))
