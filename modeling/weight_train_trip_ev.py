"""
Author : Lee Won Seok (reference 'GRU4REC_Tensorflow' ,Author : Weiping Song)
"""

import numpy as np
import pandas as pd

def evaluate_sessions_batch(model,test_data,cut_off=20,batch_size=50,*,session_key,time_key,item_key):
    '''
    Evaluates the GRU4Rec network wrt. recommendation accuracy by recall@N and MRR@N.


    :param model: A trained GRU4Rec model
    :param test_data: It contains the transactions of the test set. Columns have sessionid,timeid,itemid,attributeids
    :param cut_off: int
        the length of recommendation list
    :param batch_size:
        Number of events bundled into a batch during evaluation. Speeds up evaluation. If it is set high, the memory consumption increases. Default value is 50.
    :param session_key: String
         Header of the session ID
    :param time_key: String
        Header of the time ID
    :param item_key: String
        Header of the item ID
    :param attribute_key: String
        Header of the attribute ID
    :return: tuple
        (Recall@N,MRR@N)
    '''
    model.predict = False

    test_data.sort_values([session_key,time_key],inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique()+1,dtype=np.int32) # 데이터 세션길이 만큼 array 생성
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()

    evaluation_point_count = 0
    mrr, recall = np.array([0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    if len(offset_sessions) -1 < batch_size: # 전체 세션길이가 batch_size 보다 작은 경우
        batch_size = len(offset_sessions) - 1 # 사실상 필요없는 두줄이긴 함..
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters] # 시작세션 0~49
    end = offset_sessions[iters+1] # 끝(다음) 1~50
    in_idx = np.zeros([batch_size,len(model.attribute_key)],dtype=np.int32)
    sess_idx = np.zeros([batch_size], dtype=np.int32)  # 여행추가항1
    np.random.seed(42)

    result_total = pd.DataFrame(columns=['Session', 'Input_item', 'Out_item', 'Rec_lst'])
    while True:
        valid_mask = iters >= 0 # 이 조건에 의해 valid mask가 전부 0보다 작으면 반복문 종료
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask] # true 인 부분만 나옴
        minlen = (end[valid_mask]-start_valid).min() # 세션길이중 최소길이 가져오기
        in_idx[valid_mask] = test_data[model.attribute_key].values[start_valid]
        sess_idx[valid_mask] = test_data[model.session_key].values[start_valid]  # 여행추가항2
        in_item_idx = list(test_data[item_key].values[start_valid])

        for i in range(minlen-1):
            out_idx = test_data[model.attribute_key].values[start_valid + i +1]
            out_item_idx = list(test_data[item_key].values[start_valid + i +1])
            preds = model.predict_next_batch(iters,in_idx,batch_size)
            preds.fillna(0,inplace=True)
            in_idx[valid_mask] = out_idx
            #print(list(model.item_table.iloc[lsts,0]))
            #preds[model.rating_sort] = model.item_table[model.rating_sort]
            preds_each = []
            #return preds,out_item_idx,valid_mask
            #print(preds[model.rating_sort])
            for idx in range(batch_size):
                if valid_mask[idx] == True:
                    pr = preds[[idx]]
                    pr = pr.loc[pr[idx].nlargest(100).index, :]
                    pr2 = pr.sort_values(by=[idx], ascending=[False], inplace=False)
                    pr2 = pr2.iloc[:cut_off, :]
                    preds_each.append(list(pr2.index))

            evaluation_point_count += len(out_item_idx)
            preds_each_array = np.array(preds_each)

            result2 = pd.DataFrame(columns=['Session', 'Input_item', 'Out_item', 'Rec_lst'])
            result2['Session'] = sess_idx[valid_mask]
            result2['Rec_lst'] = preds_each
            result2['Input_item'] = in_item_idx
            result2['Out_item'] = out_item_idx
            in_item_idx = out_item_idx

            result_total = pd.concat([result_total, result2], axis=0)
            out_array = np.array(out_item_idx).reshape(len(out_item_idx), 1)
            where = np.where(preds_each_array == out_array)[1] + 1
            for j, cut in enumerate([1, 5, 10, 15, 20]):
                where2 = where[where <= cut]
                recall[j] += where2.shape[0]
                mrr[j] += sum(1 / where2)

        start = start + minlen - 1
        mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]  # 뭔가 던 간다히 할 방법은 없을까?
        for idx in mask:
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]
    return recall / evaluation_point_count, mrr / evaluation_point_count, result_total

