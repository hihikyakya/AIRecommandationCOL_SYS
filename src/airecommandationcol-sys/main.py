from .services.data import make_table
from .services import P1, P2, P3


def main():
    df_problem=make_table()
    df_userCF=P1.solve(df_problem)
    df_itemCF=P2.solve(df_problem)
    P3.solve(df_problem, df_userCF, df_itemCF)



if __name__=="__main__":
    main()