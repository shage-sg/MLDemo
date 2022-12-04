# -*- coding: utf-8 -*-import pandas as pdimport numpy as npimport jsonimport sysfrom copy import deepcopyfrom utils import entropy, ginidef all_same_class(D, column='Label'):    # 判断数据集 D 中数据是否为同一类别，即 column 列的标签相同    return D[column].unique().size == 1# 判断在A上属性是否相同def all_same_feature(D, A):    total = len(D)    for attr_name, attr_info in A.items():        if attr_info['type'] == 'descrete':            if D[attr_name].unique().size != 1:                return False        else:            for attr_val in attr_info['value']:                partial = len(D.loc[(D[attr_name] > attr_val[0]) & (D[attr_name] <= attr_val[1])])                if 0 < partial < total:                    return False    return True# 获得占多数的类别def major_class(D, column='Label'):    vals, counts = np.unique(D[column], return_counts=True)    return vals[np.argmax(counts)]def choose_best_attr(D, A, method='C4.5'):    if method == 'ID3':        entr = entropy(D['Label'])        gain = []  # 保存各分支节点的信息增益        for attr_name, attr_info in A.items():            sub_entr = 0            for attr_val in attr_info['value']:                if attr_info['type'] == 'descrete':                    Dv = D.loc[D[attr_name] == attr_val]                else:                    Dv = D.loc[(D[attr_name] > attr_val[0]) & (D[attr_name] <= attr_val[1])]                sub_entr += len(Dv) * entropy(Dv['Label']) / len(D)            gain.append(entr - sub_entr)        return [i for i in A.keys()][np.argmax(gain)]    elif method == 'C4.5':        entr = entropy(D['Label'])        gain = []        gain_ratio = []        for attr_name, attr_info in A.items():            sub_entr = 0            sub_iv = 0            for attr_val in attr_info['value']:                if attr_info['type'] == 'descrete':                    Dv = D.loc[D[attr_name] == attr_val]                else:                    Dv = D.loc[(D[attr_name] > attr_val[0]) & (D[attr_name] <= attr_val[1])]                sub_entr += len(Dv) * entropy(Dv['Label']) / len(D)                if len(Dv) != 0:                    sub_iv -= len(Dv) / len(D) * np.log2(len(Dv) / len(D))            gain.append(entr - sub_entr)            gain_ratio.append((entr - sub_entr) / (sub_iv + sys.float_info.epsilon))        gain = np.array(gain)        mask = gain >= gain.mean()  # 信息增益高于平均水平的属性所在位置        gain_ratio = np.array(gain_ratio)        idx = np.argmax(gain_ratio[mask])  # 在上述属性中选择增益率最大的        A_candidate = np.array([i for i in A.keys()])        return A_candidate[mask][idx]    elif method == 'CART':        gain = []        for attr_name, attr_info in A.items():            sub_gini = 0            for attr_val in attr_info['value']:                if attr_info['type'] == 'descrete':                    Dv = D.loc[D[attr_name] == attr_val]                else:                    Dv = D.loc[(D[attr_name] > attr_val[0]) & (D[attr_name] <= attr_val[1])]                sub_gini += len(Dv) * gini(Dv['Label']) / len(D)            gain.append(sub_gini)        return [i for i in A.keys()][np.argmin(gain)]def array2str(a):    return ' ~ '.join([str(i) for i in a])def assemble_result(D, column='Label'):    vals, counts = np.unique(D[column], return_counts=True)    idx = np.argsort(-counts)    total = sum(counts)    ratio = f'{counts[idx[0]]}/{total}'    major = 'Yes' if vals[idx[0]] == 1 else 'No'    return f'{major} ({ratio})'def grow_tree(tree, D, A, method):    if all_same_class(D) or len(A) == 0 or all_same_feature(D, A):        tree['Result'] = assemble_result(D)        return    best_attr = choose_best_attr(D, A, method=method)    tree[best_attr] = {}    attr_info = A[best_attr]    for attr_val in attr_info['value']:        if attr_info['type'] == 'descrete':            Dv = D.loc[D[best_attr] == attr_val]        else:            Dv = D.loc[(D[best_attr] > attr_val[0]) & (D[best_attr] <= attr_val[1])]        branch = f'{attr_val}' if attr_info['type'] == 'descrete' else array2str(attr_val)        if len(Dv) == 0:            tree[best_attr][branch] = {}            tree[best_attr][branch]['Result'] = assemble_result(D) + ' by parent node'        else:            tree[best_attr][branch] = {}            A_ = deepcopy(A)            A_.pop(best_attr)            grow_tree(tree[best_attr][branch], Dv, A_, method=method)def evaluate_tree(tree, D, A):    for node, branch in tree.items():        if node == 'Result':            return 0 if branch.startswith('No') else 1        else:            if A[node]['type'] == 'descrete':                subtree = tree[node][str(D[node])]            else:                for attr_val in A[node]['value']:                    if attr_val[0] < D[node] <= attr_val[1]:                        branch = array2str(attr_val)                        subtree = tree[node][branch]                        break            return evaluate_tree(subtree, D, A)if __name__ == '__main__':    # 用周志华《机器学习》中的西瓜数据分类结果检验代码（ID3算法生成的决策树在原数78页）    D = pd.read_csv('waterMelon2.0.csv', sep=',')    del D['编号']    D.loc[(D['好瓜'] == '是'), '好瓜'] = 1    D.loc[(D['好瓜'] == '否'), '好瓜'] = 0    D = D.rename(columns={'好瓜': 'Label'})    A = {        "色泽": {'type': 'descrete', 'value': ["青绿", "乌黑", "浅白"]},        "根蒂": {'type': 'descrete', 'value': ["蜷缩", "稍蜷", "硬挺"]},        "敲声": {'type': 'descrete', 'value': ["浊响", "沉闷", "清脆"]},        "纹理": {'type': 'descrete', 'value': ["清晰", "稍糊", "模糊"]},        "脐部": {'type': 'descrete', 'value': ["凹陷", "稍凹", "平坦"]},        "触感": {'type': 'descrete', 'value': ["硬滑", "软粘"]}    }    tree = {}    grow_tree(tree, D, A, method='ID3')    pretty = json.dumps(tree, indent=4, ensure_ascii=False)    print(pretty)