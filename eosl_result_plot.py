from modified_eosl import eosl_loss
import pandas as pd
import matplotlib.pyplot as plt


def get_eosl_stats(plot_graph_option=0, default_weight=1):
    enc_list = ['vit', 'gitbase', 'blipbase', 'gitlarge', 'bliplarge']
    if plot_graph_option:
        i = 0
        dfs = {}

        for enc in enc_list:
            df = pd.read_csv('eosl_plotdata.csv')
            df_data = df[df['encoder'] == enc]
            dfs[i] = df_data
            i += 1
        # Plot dataset 1 as a line plot
        plt.plot(dfs[0]['pb'], dfs[0]['eosl'], label='vit', marker='x')
        plt.xscale('log')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[1]['pb'], dfs[1]['eosl'], label='gitbase', marker='+')
        plt.xscale('log')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[2]['pb'], dfs[2]['eosl'], label='blipbase', marker='*')
        plt.xscale('log')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[3]['pb'], dfs[3]['eosl'], label='gitlarge', marker='o')
        plt.xscale('log')

        # Plot dataset 2 as a line plot
        plt.plot(dfs[4]['pb'], dfs[4]['eosl'], label='bliplarge', marker='p')
        plt.xscale('log')

        # Add labels, title, and legend
        legend_font = {'weight': 'bold', 'size': 12}
        plt.xlabel('Probability of Bit Error', fontweight='bold', size=12)
        plt.ylabel('Energy Optimized Sematic Loss', fontweight='bold', size=12)
        plt.title('EOSL v bit error probability for various encoders', fontweight='bold', size=12)
        plt.legend(loc='lower right', frameon=False, prop=legend_font)

        # Show the plot
        plt.show()

    else:
        plot_data_all = []
        # pb_list = [.0002, .001, .005, .025, .125]
        pb_list = [i / 1000 for i in range(1, 101)]
        plot_data_all.append(['encoder', 'pb', 'eosl'])

        if default_weight:
            for enc in enc_list:
                for pb in pb_list:
                    plot_data_all.append([enc, pb, eosl_loss(enc, pb)])

        if not default_weight:
            print('------please enter the weightage values for eosl-------\n')
            lambda_es = float(input('enter the weight for semantic energy loss:lambda_es\n'))
            lambda_sm = float(input('enter the weight for semantic dissimilarity:lambda_sm\n'))
            lambda_lch = float(input('enter the weight for chanel loss:lambda_lch\n'))
            lambda_ec = float(input('enter the weight for comm. energy loss:lambda_ec\n'))

            for enc in enc_list:
                for pb in pb_list:
                    plot_data_all.append([enc, pb, eosl_loss(enc, pb, lambda_es, lambda_sm, lambda_lch, lambda_ec)])

        plot_data_panda = pd.DataFrame(plot_data_all)
        plot_data_panda.to_csv('eosl_plotdata.csv', index=False, header=False)


if __name__ == '__main__':
    plot_graph = int(input('add new plotdata or create graph: 1 for graph, 0 for else:\n'))
    default = 1
    if not plot_graph:
        default = int(input('use default weights or customized: 1 for default, 0 for else:\n'))
    get_eosl_stats(plot_graph, default)
