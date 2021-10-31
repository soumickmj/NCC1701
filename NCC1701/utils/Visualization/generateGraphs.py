import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def generatePlot(data, x, y, hue, type='violin', palette="pastel", violin_inner='stick', violin_split=True, swarm_as_inner=True, subplot_col_cat=None, subplot_col_row=None):
    #type: violin (default) ¦ box ¦ boxen
    #violin_inner [only applicable to violin plots]: box (default) ¦ quartile ¦ point ¦ stick ¦ None 
    #violin_split [do]: if set to True, then two data gets merged into 1 violin (ideal to show comparison between output and groundtruth)
    #swarm_as_inner [can be used with all, but should only be used with violin]: sets whether to plot all data point individually inside the main plot as a swarm plot
    #split_by_col: if want to split into sub-graphs, based on a certain column
    if swarm_as_inner and subplot_col_cat is None and subplot_col_row is None: #TODO: Fix problem with using subplot along with swarm
        violin_inner=None
    if type == 'violin':
        g = sns.catplot(x=x, y=y, hue=hue,col=subplot_col_cat,row=subplot_col_row,
                kind=type, inner=violin_inner, split=violin_split,  
                palette=palette, data=data)
    else:
        g = sns.catplot(x=x, y=y, hue=hue,col=subplot_col_cat,row=subplot_col_row,
                kind=type, palette=palette, data=data)
    if swarm_as_inner and subplot_col_cat is None and subplot_col_row is None:
        sns.swarmplot(x=x, y=y, color="k", size=3, data=data,  ax=g.ax)

    #To remove some side of the plot, for example left side

    #for ax in g.axes.flat:
    #    sns.despine(left=True,ax=ax)
    return g

#tips = sns.load_dataset("tips")
##tips=tips.drop(columns='tip')
#generatePlot(tips, 'day','total_bill','sex', type='violin')
#plt.show()

def generatePlotFromExcel(excel_path, matrix='SSIM'):
    #matrix: SSIM ¦ MSE ¦ RMSE ¦ PSNR ¦ SSD
    df = pd.read_excel(excel_path, header=[1,2], skipfooter=1)
    averages = df.mean(axis=0)
    #Choose which version to use (in case there are multiple)
    if matrix == 'SSIM' or matrix == 'PSNR':
        type1 = averages[matrix].idxmax(axis=0)
        type2 = 'Under'
    else:
        type1 = 'Recon'
        type2 = 'Under'
    x = df[['Unnamed: 0_level_0', matrix]]
    x.columns = x.columns.droplevel()
    x = x.melt(id_vars=["FileName"], 
            var_name="Type", 
            value_name="Value")
    x = x[x['Type'].isin([type1,type2])]
    x = x.replace(type1,'Recon')
    x['test_type'] = 1 
    generatePlot(x, 'test_type','Value','Type', type='violin', violin_split=True, swarm_as_inner=False, violin_inner=None)

def generateFrameFromExcel(excel_path, matrix='SSIM', group_name=None, test_name=None):
    #matrix: SSIM ¦ MSE ¦ RMSE ¦ PSNR ¦ SSD
    df = pd.read_excel(excel_path, header=[1,2], skipfooter=1)
    averages = df.mean(axis=0)
    #Choose which version to use (in case there are multiple)
    if matrix == 'SSIM' or matrix == 'PSNR':
        type = averages[matrix].idxmax(axis=0)
    else:
        type = 'Recon'
    df = df[['Unnamed: 0_level_0', matrix]]
    df.columns = df.columns.droplevel()
    df = df.melt(id_vars=["FileName"], 
            var_name="Type", 
            value_name="Value")
    df = df[df['Type'].isin([type])]
    if group_name is not None:
        df = df.replace(type,group_name)
    df['test_type'] = test_name
    
    return df
    


def generatePlotFromAnalysisPathExcel(excel_path, matrix='SSIM'):
    analysisPaths = pd.read_excel(excel_path, header=0)
    frames = None
    for index, row in analysisPaths.iterrows():
        test_name = row[0]
        group_frames = None
        n_groups = len(row)-1
        for i in range(1, len(row)):
            group_name = row.index[i]
            df = generateFrameFromExcel(row[i], matrix=matrix, group_name=group_name, test_name=test_name)
            if group_frames is None:
                group_frames = df
            else:
                #group_frames = group_frames.join(df, on='Type')
                group_frames = pd.concat([group_frames, df])
        if frames is None:
            frames = group_frames
        else:
            #frames = frames.join(group_frames, on='test_type')
            frames = pd.concat([frames, group_frames])

    if n_groups == 2:
        violin_split = True
    else:
        violin_split = False
    generatePlot(frames, 'test_type','Value','Type', type='violin', violin_split=violin_split, swarm_as_inner=False, violin_inner=None)


#generatePlotFromExcel(r'D:\\Output\\Gen3\\IXI-T1\\HH\\AllSlices\\Attempt7-IXIT1HHVarden1D30-Resnet2Dv214PReLU-SSIMLoss-AllSlices-Combined\\Analysis.xlsx', matrix='SSIM')
generatePlotFromAnalysisPathExcel(r"E:\Output\AnalysisPaths.xlsx")
plt.show()