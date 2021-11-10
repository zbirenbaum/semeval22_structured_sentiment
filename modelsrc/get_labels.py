def get_opinion_types():
    df1 = pd.DataFrame(data)
    list_opinion_types = []
    for _, row in df1.iterrows():
        for i in range(len(row['opinions'])):
            opinion = row['opinions'][i]
            for key in dict(opinion).keys():
                list_opinion_types.append(key)
    return list(set(list_opinion_types))

    #for opinions in datapoint['opinions']:
    #    print(opinions.keys())
    
#print(init[10])
#    opinions = row['opinions']
opinion_labels = get_opinion_types()
