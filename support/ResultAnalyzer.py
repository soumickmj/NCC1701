import os
import xlsxwriter

def AnalyzeResultFolder(root, folder, isFolderInFolder=True, isAnyCorrected=False):
    root_path = os.path.join(root, folder)
    subfolders = [f for f in os.listdir(root_path) if f != 'checkpoints' and not os.path.isfile(os.path.join(root_path, f))] #Get all subfolders (subject specific folders) from root
    workbook_path =  os.path.join(root_path, 'Analysis.xlsx')
    workbook = xlsxwriter.Workbook(workbook_path)
    worksheet = workbook.add_worksheet(folder.split('-')[0])
    title_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'})
    worksheet.merge_range('A1:J1', folder, title_format)
    row = 2
    col = 0 
    worksheet.write(row, 0, 'FileName', title_format)
    if(isAnyCorrected):
        worksheet.merge_range('B2:G2', 'SSIM', title_format)
        worksheet.merge_range('H2:J2', 'MSE', title_format)
        worksheet.merge_range('K2:M2', 'RMSE', title_format)
        col=3 #moved 3 additional positions, becuase we are adding three new columns
        worksheet.write(row, col-2, 'B4Cor', title_format)
        worksheet.write(row, col-1, 'AftrCor', title_format)
        worksheet.write(row, col, 'AftrNormCor', title_format) 
    else:
        worksheet.merge_range('B2:D2', 'SSIM', title_format)
        worksheet.merge_range('E2:G2', 'MSE', title_format)
        worksheet.merge_range('H2:J2', 'RMSE', title_format)
    worksheet.write(row, col+1, 'Recon', title_format)
    worksheet.write(row, col+2, 'Under', title_format)
    worksheet.write(row, col+3, 'Improve', title_format)
    worksheet.write(row, col+4, 'Recon', title_format)
    worksheet.write(row, col+5, 'Under', title_format)
    worksheet.write(row, col+6, 'Improve', title_format)
    worksheet.write(row, col+7, 'Recon', title_format)
    worksheet.write(row, col+8, 'Under', title_format)
    worksheet.write(row, col+9, 'Improve', title_format)
    worksheet.autofilter('A3:J3')

    mse = []
    rmse = []
    ssim = []
    mseUnder = []
    rmseUnder = []
    ssimUnder = []
    mseImprove = []
    rmseImprove = []
    ssimImprove = []
    ssimBeforeCorrecAll = []
    ssimAfterCorrecAll = []
    ssimAfterNormCorrecAll = []

    for folder in subfolders:   
        print(folder)
        fullpath_subfolder = os.path.join(root_path, folder)
        if(isFolderInFolder):
            folderInfolder = [f for f in os.listdir(fullpath_subfolder) if not os.path.isfile(os.path.join(fullpath_subfolder, f))] #Get all subfolders (subject specific folders) from root
            fullpath_subfolder = os.path.join(fullpath_subfolder, folderInfolder[0])
        accuracy = os.path.join(fullpath_subfolder, 'accuracy.txt')   
        accuracyOfUndersampled = os.path.join(fullpath_subfolder, 'accuracyOfUndersampled.txt')          
        improvement = os.path.join(fullpath_subfolder, 'improvement.txt')  
        afterDataConsistancy = os.path.join(fullpath_subfolder, 'afterDataConsistancy.txt') 
        afterDataConsistancyNorm = os.path.join(fullpath_subfolder, 'afterDataConsistancyNorm.txt')

        F_accuracy = open(accuracy,"r") 
        F_accuracy_data = F_accuracy.readlines()
        F_accuracy.close()
        F_accuracy_data = list(filter(None, map(lambda s: s.strip(), F_accuracy_data)))
        mse.append(float(F_accuracy_data[1].strip().split(":")[1].strip()))
        rmse.append(float(F_accuracy_data[2].strip().split(":")[1].strip()))
        ssim.append(float(F_accuracy_data[3].strip().split(":")[1].strip()))

        F_accuracyUnder = open(accuracyOfUndersampled,"r") 
        F_accuracyUnder_data = F_accuracyUnder.readlines()
        F_accuracyUnder.close()
        F_accuracyUnder_data = list(filter(None, map(lambda s: s.strip(), F_accuracyUnder_data)))
        mseUnder.append(float(F_accuracyUnder_data[1].strip().split(":")[1].strip()))
        rmseUnder.append(float(F_accuracyUnder_data[2].strip().split(":")[1].strip()))
        ssimUnder.append(float(F_accuracyUnder_data[3].strip().split(":")[1].strip()))

        F_accuracyImprove = open(improvement,"r") 
        F_accuracyImprove_data = F_accuracyImprove.readlines()
        F_accuracyImprove.close()
        F_accuracyImprove_data = list(filter(None, map(lambda s: s.strip(), F_accuracyImprove_data)))
        mseImprove.append(float(F_accuracyImprove_data[1].strip().split(":")[1].strip()))
        rmseImprove.append(float(F_accuracyImprove_data[2].strip().split(":")[1].strip()))
        ssimImprove.append(float(F_accuracyImprove_data[3].strip().split(":")[1].strip()))

        if(isAnyCorrected):
            try:
                F_accuracyAfterDConsis = open(afterDataConsistancy,"r") 
                F_accuracyAfterDConsis_data = F_accuracyAfterDConsis.readlines()
                F_accuracyAfterDConsis.close()
                F_accuracyAfterDConsis_data = list(filter(None, map(lambda s: s.strip(), F_accuracyAfterDConsis_data)))
                ssimBeforeCorrec = float(F_accuracyAfterDConsis_data[0].strip().split(":")[1].strip())
                ssimBeforeCorrecAll.append(ssimBeforeCorrec)
                ssimAfterCorrec = float(F_accuracyAfterDConsis_data[1].strip().split(":")[1].strip())
                ssimAfterCorrecAll.append(ssimAfterCorrec)
            except FileNotFoundError:
                ssimBeforeCorrec = None
                ssimAfterCorrec = None
            try:
                F_accuracyAfterDConsisNorm = open(afterDataConsistancyNorm,"r") 
                F_accuracyAfterDConsisNorm_data = F_accuracyAfterDConsisNorm.readlines()
                F_accuracyAfterDConsisNorm.close()
                F_accuracyAfterDConsisNorm_data = list(filter(None, map(lambda s: s.strip(), F_accuracyAfterDConsisNorm_data)))
                ssimAfterNormCorrec = float(F_accuracyAfterDConsisNorm_data[1].strip().split(":")[1].strip())
                ssimAfterNormCorrecAll.append(ssimAfterNormCorrec)
            except FileNotFoundError:
                ssimAfterNormCorrec = None

        
        row += 1
        worksheet.write(row, 0, folder)
        if(isAnyCorrected):
            worksheet.write(row, col-2, ssimBeforeCorrec)
            worksheet.write(row, col-1, ssimAfterCorrec)
            worksheet.write(row, col, ssimAfterNormCorrec) 
        worksheet.write(row, col+1, ssim[-1])
        worksheet.write(row, col+2, ssimUnder[-1])
        worksheet.write(row, col+3, ssimImprove[-1])
        worksheet.write(row, col+4, mse[-1])
        worksheet.write(row, col+5, mseUnder[-1])
        worksheet.write(row, col+6, mseImprove[-1])
        worksheet.write(row, col+7, rmse[-1])
        worksheet.write(row, col+8, rmseUnder[-1])
        worksheet.write(row, col+9, rmseImprove[-1])

    worksheet.write(row, 0, 'Average', title_format)
    if(isAnyCorrected):
        try:
            worksheet.write(row, col-2, sum(ssimBeforeCorrecAll)/len(ssimBeforeCorrecAll), title_format)
        except:
            worksheet.write(row, col-2, 0, title_format)
        try:
            worksheet.write(row, col-1, sum(ssimAfterCorrecAll)/len(ssimAfterCorrecAll), title_format)
        except:
            worksheet.write(row, col-1, 0, title_format)
        try:
            worksheet.write(row, col, sum(ssimAfterNormCorrecAll)/len(ssimAfterNormCorrecAll), title_format) 
        except:
            worksheet.write(row, col, 0, title_format) 
    worksheet.write(row, col+1, sum(ssim)/len(ssim), title_format)
    worksheet.write(row, col+2, sum(ssimUnder)/len(ssimUnder), title_format)
    worksheet.write(row, col+3, sum(ssimImprove)/len(ssimImprove), title_format)
    worksheet.write(row, col+4, sum(mse)/len(mse), title_format)
    ##worksheet.write(row, col+5, sum(mseUnder)/len(mseUnder), title_format)
    worksheet.write(row, col+6, sum(mseImprove)/len(mseImprove), title_format)
    worksheet.write(row, col+7, sum(rmse)/len(rmse), title_format)
    worksheet.write(row, col+8, sum(rmseUnder)/len(rmseUnder), title_format)
    worksheet.write(row, col+9, sum(rmseImprove)/len(rmseImprove), title_format)

    workbook.close()

    


#AnalyzeResultFolder(r'D:\Output\Attempt20-WithValidate-OASISVarden30SameMask-DenseResNetSSIMLoss-SingleSlice50', 'epoch200', True, True)