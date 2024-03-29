from modified_eosl import eosl_loss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
        plt.xlabel('Probability of Bit Error', fontweight='bold', size=14)
        plt.ylabel('Energy Optimized Sematic Loss', fontweight='bold', size=14)
        plt.title('EOSL v bit error probability for various encoders', fontweight='bold', size=14)
        plt.legend(loc='best', frameon=False, prop=legend_font)
        # Below title is for default values of weights, change it manually as per the weights
        # plt.title(r'$\mathbf{(\lambda_{\mathrm{es}},\ \lambda_{\mathrm{sm}},\ \lambda_{\mathrm{lch}},\ \lambda_{'
        #           r'\mathrm{ec}})}$ = 1', fontweight='bold', fontsize=14)

        # title for lambda > 1

        plt.title(r'$\mathbf{(\lambda_{\mathrm{sm}},\ \lambda_{\mathrm{lch}},\ \lambda_{\mathrm{ec}})}$ = 1, '
                  r'$\mathbf{\lambda_{\mathrm{es}}}$ = 10', fontweight='bold', fontsize=14)

        # title for lambda < 1
        # plt.title(r'$\mathbf{(\lambda_{\mathrm{es}},\ \lambda_{\mathrm{lch}},\ \lambda_{\mathrm{sm}})}$ = 1, '
        #           r'$\mathbf{\lambda_{\mathrm{ec}}}$ = 0.1', fontsize=12)
        # Show the plot
        plt.show()

    else:
        plot_data_all = []
        ranges = np.arange(-30, -10, 1)
        pb_list = [pow(10, i / 10) for i in ranges]
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
