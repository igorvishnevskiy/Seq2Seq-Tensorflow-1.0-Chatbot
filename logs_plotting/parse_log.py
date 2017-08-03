__author__ = 'ivishnevskiy'


import matplotlib.pyplot as plt
import numpy as np



class PlotLogs():


    def parse_logs(self):
        filename = "logs.txt"
        with open(filename, "r") as logs_file:
            logs_out = logs_file.read()
        logs_parts = logs_out.split("global step")
        steps_out = []
        p_avg_out = []
        glob_p_out = []
        l_rate_out = []

        for part in logs_parts:
            additional_split = part.split("perplexity")
            prplxt_container = []
            avrg_prplxt_container = []
            step = 0
            learning_rate = 0.0
            for idx, add_part in enumerate(additional_split):
                if len(add_part) > 0:
                    if idx == 0:
                        global_part = add_part.split("learning rate")
                        step = float(global_part[0].strip())
                        sec_part = global_part[1].split("step-time")
                        learning_rate = float(sec_part[0].strip())
                    else:
                        prplxt_value = float(add_part.split("\n")[0].strip())
                        prplxt_container.append(prplxt_value)
                        avrg_prplxt_container.append(prplxt_value)
                        if idx == 1:
                            glob_p_out.append(prplxt_value)
            if len(prplxt_container) > 0:
                # print step
                steps_out.append(step)
                # print learning_rate
                l_rate_out.append(learning_rate)
                # print prplxt_container
                # print sum(prplxt_container)
                # print len(prplxt_container)
                perplexity_average = sum(avrg_prplxt_container)/len(avrg_prplxt_container)
                # print perplexity_average
                p_avg_out.append(perplexity_average)

        # print p_avg_out.index(min(p_avg_out))
        # print steps_out
        # print p_avg_out
        # print glob_p_out
        # print l_rate_out
        self.create_plot(steps_out, p_avg_out, "perplexity_average", "Average Perplexity")
        self.create_plot(steps_out, glob_p_out, "global_perplexity", "Global Perplexity")
        self.create_plot(steps_out, l_rate_out, "learning_rate", "Learning Rate")



    def create_plot(self, x, y, plot_name, ylabel):
        fig = plt.figure(figsize=(11,8))
        ax1 = fig.add_subplot(111)

        plt.xlabel("Steps")
        plt.ylabel(ylabel)

        ax1.plot(x, y)
        ax1.legend(loc=2)

        colormap = plt.cm.gist_ncar
        colors = [colormap(i) for i in np.linspace(0, 1,len(ax1.lines))]
        for i,j in enumerate(ax1.lines):
            j.set_color(colors[i])

        plt.savefig(str(plot_name)+".png")



run = PlotLogs()
run.parse_logs()
