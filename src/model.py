import numpy as np
import pandas as pd

class Model:

    def place_bets(
        self,
        summary: pd.DataFrame,
        opps: pd.DataFrame,
        inc: pd.DataFrame,
    ):
        
      
        def clear_data(data: pd.DataFrame) -> pd.DataFrame:
            import pandas as pd
            import numpy as np

            
            """pd.reset_option('display.max_rows')
            pd.set_option('display.max_columns', None)"""

             
            df = pd.read_csv("data/games.csv")
            #df.head()


        

            df[["OddsH", "OddsA"]] = df[["OddsH", "OddsA"]].replace(0, np.nan)


             
            df = df.dropna(subset=['OddsH'])

             
            df = df.drop("Unnamed: 0", axis = 1)

             
            #df.head() #dropping null values for odds because we need them for our model

             
            #df.info()

             
            df = df.drop(columns = ["Date"]) #drop dates of games
            """print('Date' in df.columns)
            df.info()"""

             
            """(df['D'] == True).sum()
            df.info()"""

             
            """draw_list = df.loc[df["D"]]
            df.loc[(df['D'] == True) & (df["Special"].notna()) & (df["H_SO"].notna())] # draw is not clasified as winning in nájezdy
            """

             
            """df.loc[(df['D'] == True) & (df["Special"].notna())] #draws are classified as draw in over time circa 100/500"""

             
            """df.loc[(df['D'] == True)] #& (df["H_SO"].notna())] #aroud 400 draws with no overtime/ NO data on DRAW AND THEN NAJEZDY, #hypothesis: is draw divided with some year (OT and no OT?)"""

             

            #hypothesis: advanced features where measured after some year # there is 236 games where there has been actually najezdy and overtime

             
            """df.loc[df["H_SOG"].notna()]"""

             
            """count = (df.loc[6904:, 'H_SOG'].isna()).sum() #there is sth wrong, because they prob started with new year but hadnt done it perfectly
            print(count)"""

             
            """df.loc[df["H_BLK_S"].notna()]
            change_of_features = 6970"""

             
            """count = (df.loc[change_of_features:, 'H_BLK_S'].isna()).sum() #hypothesis confirmed: season 2009 was start of advanced features
            #split data in this year and do ranksum test separately? yes -> see results and and conclude from rank sum test
            print(count)"""

             
            #použit median misto nan pro svm metodu
            #A_SOG	H_BLK_S	A_BLK_S	H_HIT	A_HIT	H_BLK	A_BLK	H_FO	A_FO
            median_H_SOG = df['H_SOG'].median()
            median_A_SOG = df["A_SOG"].median()
            median_H_BLK_S = df['H_BLK_S'].median()
            median_A_BLK_S = df['A_BLK_S'].median()
            median_H_HIT = df['H_HIT'].median()
            median_A_HIT = df['A_HIT'].median()
            median_H_BLK = df['H_BLK'].median()
            median_A_BLK = df['A_BLK'].median()
            median_H_FO = df['H_FO'].median()
            median_A_FO = df['A_FO'].median()

            df["H_SOG"] = df["H_SOG"].fillna(median_H_SOG)
            df["A_SOG"] = df["A_SOG"].fillna(median_A_SOG)
            df["H_BLK_S"] = df["H_BLK_S"].fillna(median_A_BLK_S)
            df["A_BLK_S"] = df["A_BLK_S"].fillna(median_A_BLK_S)
            df["H_HIT"] = df["H_HIT"].fillna(median_H_HIT)
            df["A_HIT"] = df["A_HIT"].fillna(median_A_HIT)
            df["H_BLK"] = df["H_BLK"].fillna(median_H_BLK)
            df["A_BLK"] = df["A_BLK"].fillna(median_A_BLK)
            df["H_FO"] = df["H_FO"].fillna(median_H_FO)
            df["A_FO"] = df["A_FO"].fillna(median_A_FO)

            #df.info()

             
            #f.loc[(df['H_PPG'].isna())] #cca 20 zapásu chybí udáj o golech v přesilovce, a všechny mají draw true -> 0
            df['H_PPG'] = df['H_PPG'].fillna(0)
            df['A_PPG'] = df['A_PPG'].fillna(0)
            df["H_SHG"] = df["H_SHG"].fillna(0)
            df["A_SHG"] = df["A_SHG"].fillna(0)

             
            #df.info()

             
            H_SV_median = df['H_SV'].median()
            A_SV_median = df['A_SV'].median()
            df['H_SV'] = df['H_SV'].fillna(H_SV_median)
            df['A_SV'] = df['A_SV'].fillna(A_SV_median)
            #df.info()

             
            #df.loc[df["H_P3"].isna()]

             
            #tohle musím dopnit ručně, aby to sedělo s počtem golu a special eventem, udělám to asi uniformě
            df = df.dropna(subset=['H_P3', 'A_P3'])
            df = df.dropna(subset=['H_P2', 'A_P2'])



             
            #check whether HSO are only when special is PS
            #df.loc[(df['H_SO'].notna()) & (df['Special'] == 'PS')]
            #its ok, sth is contumacy (3 games circa) 
            df["H_SO"] = df["H_SO"].fillna(0)
            df["A_SO"] = df["A_SO"].fillna(0)
            #df.info()


             
            #df.loc[(df['H_OT'].notna()) & (df['Special'].isna())] # 1 game and its ok, other games (notna and special ot + ps and dw) are also ok -> all h_OT not null = 0
            #its ok, sth is contumacy (3 games circa) 
            df["H_OT"] = df["H_OT"].fillna(0)
            df["A_OT"] = df["A_OT"].fillna(0)
            #df.info()



             
            #and special add B jako basic game and use dummy var
            df["Special"] = df["Special"].fillna("B")
            #df.info()

             
            #drop Open column
            df = df.drop("Open", axis=1)
            df.fillna(0, inplace=True)
            #df.head()

             
            df["H"] = df["H"].astype('category')
            df["A"] = df["A"].astype('category')
            df["D"] = df["D"].astype('category')
            special = df[["Special"]].drop_duplicates().reset_index(drop=True)
            #display(special)

             
            special_category = pd.api.types.CategoricalDtype(categories=special.Special.values, ordered=True)
            df["Special"] = df["Special"].astype(special_category)
            

             
            df[df.select_dtypes(['category']).columns] = df.select_dtypes(['category']).apply(lambda x: x.cat.codes)
            



    



            Q1 = df["H_PIM"].quantile(0.25)
            Q3 = df["H_PIM"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df["H_PIM"] = df["H_PIM"].clip(upper= upper)


            Q1 = df["A_PIM"].quantile(0.25)
            Q3 = df["A_PIM"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df["A_PIM"] = df["A_PIM"].clip(upper= upper)


            Q1 = df["H_SV"].quantile(0.25)
            Q3 = df["H_SV"].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5*IQR
            upper = Q3 + 1.5*IQR

            df["H_SV"] = df["H_SV"].clip(upper= upper)


            df["H_MAJ"] = df["H_MAJ"].clip(upper= 1)


            df["A_MAJ"] = df["A_MAJ"].clip(upper= 1)


             
            return df

        





        min_bet = summary.iloc[0]["Min_bet"]
        N = len(opps)

        bets = np.zeros((N, 3))
        bets[np.arange(N), np.random.choice([0, 1, 2])] = min_bet
        bets = pd.DataFrame(
            data=bets, columns=["BetH", "BetA", "BetD"], index=opps.index
        )
        return bets
