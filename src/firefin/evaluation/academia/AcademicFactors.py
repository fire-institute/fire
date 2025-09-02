from __future__ import annotations
import pandas as pd
from typing import Tuple, List
from ...core.algorithm.portfolio_sort import PortfolioSort


class AcademicFactors:

    @staticmethod
    def _check_align(*dfs: pd.DataFrame) -> None:
        idx0, cols0 = dfs[0].index, dfs[0].columns
        for d in dfs[1:]:
            assert idx0.equals(d.index) and cols0.equals(d.columns), \
                "Input DataFrames must share identical index/columns"

    @staticmethod
    def _vw_market_return(stock_return: pd.DataFrame, market_cap: pd.DataFrame,lag_weights: bool = True) -> pd.Series:
        AcademicFactors._check_align(stock_return, market_cap)
        mcap = market_cap.shift(1) if lag_weights else market_cap
        w = mcap.div(mcap.sum(axis=1), axis=0)
        mkt = (w * stock_return).sum(axis=1)
        return mkt  # Series

    @staticmethod
    def _two_by_three_hl_avg(
            return_adj: pd.DataFrame,
            size_first: pd.DataFrame,
            x_second: pd.DataFrame,
            market_cap: pd.DataFrame | None,
            quantiles: Tuple[int, int] = (2, 3),
            value_weighted: bool = True,
            dependent: bool = True, 
    ) -> pd.Series:

        if value_weighted:
            assert market_cap is not None, "market_cap is required for value-weighted returns"

        qr = PortfolioSort.double_sort_return_adj_academic(
            size=size_first,
            x=x_second,
            return_adj=return_adj,
            value_weighted=value_weighted,
            market_cap=market_cap,
        )

        df = qr[0]

        s_low, s_high = "1_1", "1_3"
        b_low, b_high = "2_1", "2_3"
        df_needed = df.reindex(columns=[s_high, s_low, b_high, b_low])
        series = 0.5 * ((df_needed[s_high] - df_needed[s_low]) + (df_needed[b_high] - df_needed[b_low]))
        return series.astype(float)

    @staticmethod
    def _two_by_three_diff(
        return_adj: pd.DataFrame,
        size_first: pd.DataFrame,
        x_second: pd.DataFrame,
        market_cap: pd.DataFrame | None,
        quantiles: Tuple[int, int] = (2, 3),
        value_weighted: bool = True,
        dependent: bool = True,
        diff: str = "H-L",  
    ) -> pd.Series:
       
        core = AcademicFactors._two_by_three_hl_avg(
            return_adj=return_adj,
            size_first=size_first,
            x_second=x_second,
            market_cap=market_cap,
            quantiles=quantiles,
            value_weighted=value_weighted,
            dependent=dependent,
        )
        return core if diff.upper() == "H-L" else -core

    @staticmethod
    def _smb_from_double_sort_df(df: pd.DataFrame, quantiles: Tuple[int, int] = (2, 3)) -> pd.Series:
      
        q1, q2 = quantiles
        s_cols = [f"1_{j}" for j in range(1, q2 + 1)]
        b_cols = [f"{q1}_{j}" for j in range(1, q2 + 1)]
        df_needed = df.reindex(columns=s_cols + b_cols)
        smb = df_needed[s_cols].mean(axis=1) - df_needed[b_cols].mean(axis=1)
        return smb  # Series

    @staticmethod
    def market_excess(
        stock_return: pd.DataFrame,
        market_cap: pd.DataFrame,
        risk_free_rate: pd.Series | pd.DataFrame,
    ) -> pd.Series:
        
        mkt_vw = AcademicFactors._vw_market_return(stock_return, market_cap)
        if isinstance(risk_free_rate, pd.DataFrame):
            assert risk_free_rate.shape[1] == 1, "risk_free_rate DataFrame must be 1-column"
            rf = risk_free_rate.iloc[:, 0].reindex(stock_return.index)
        else:
            rf = risk_free_rate.reindex(stock_return.index)
        out = (mkt_vw - rf).astype(float)
        out.name = "MKT"
        return out


    @staticmethod
    def smb_ff3(
            stock_return: pd.DataFrame,
            size: pd.DataFrame,
            book_to_market: pd.DataFrame,
            market_cap: pd.DataFrame | None = None,
            quantiles: Tuple[int, int] = (2, 3),  
            value_weighted: bool = True,
    ) -> pd.Series:
        """
       SMB = mean(S/L,S/M,S/H) − mean(B/L,B/M,B/H)。
        """
        AcademicFactors._check_align(stock_return, size, book_to_market)
        if value_weighted:
            assert market_cap is not None, "market_cap is required for value-weighted SMB"
            AcademicFactors._check_align(stock_return, market_cap)

        qr = PortfolioSort.double_sort_return_adj_academic(
            size=size,
            x=book_to_market,
            return_adj=stock_return,
            value_weighted=value_weighted,
            market_cap=market_cap,
        )
        df = qr[0]  # Time × groups: '1_1','1_2','1_3','2_1','2_2','2_3'
        out = AcademicFactors._smb_from_double_sort_df(df, quantiles=(2, 3)).astype(float)
        out.name = "SMB"
        return out

    @staticmethod
    def smb_ff5(
            stock_return: pd.DataFrame,
            size: pd.DataFrame,
            book_to_market: pd.DataFrame,
            profitability: pd.DataFrame,
            investment: pd.DataFrame, 
            market_cap: pd.DataFrame | None = None,
            quantiles: Tuple[int, int] = (2, 3), 
            value_weighted: bool = True,
    ) -> pd.Series:
        AcademicFactors._check_align(stock_return, size, book_to_market, profitability, investment)
        if value_weighted:
            assert market_cap is not None, "market_cap is required for value-weighted SMB"
            AcademicFactors._check_align(stock_return, market_cap)

        parts: List[pd.Series] = []
        for x in (book_to_market, profitability, investment):
            qr = PortfolioSort.double_sort_return_adj_academic(
                size=size,
                x=x,
                return_adj=stock_return,
                value_weighted=value_weighted,
                market_cap=market_cap,
            )
            df = qr[0]
            parts.append(AcademicFactors._smb_from_double_sort_df(df, quantiles=(2, 3)))

        out = pd.concat(parts, axis=1).mean(axis=1, skipna=True).astype(float)
        out.name = "SMB"
        return out

    @staticmethod
    def hml(
        stock_return: pd.DataFrame,
        size: pd.DataFrame,
        book_to_market: pd.DataFrame,
        market_cap: pd.DataFrame | None = None,
        quantiles: Tuple[int, int] = (2, 3),
        value_weighted: bool = True,
    ) -> pd.Series:
        """
        HML：0.5×[(S,H−L) + (B,H−L)] (SH+BH)/2 − (SL+BL)/2。
        """
        out = AcademicFactors._two_by_three_hl_avg(
            return_adj=stock_return,
            size_first=size,
            x_second=book_to_market,  
            market_cap=market_cap,
            quantiles=quantiles,
            value_weighted=value_weighted,
            dependent=True,
        ).astype(float)
        out.name = "HML"
        return out

    @staticmethod
    def rmw(
        stock_return: pd.DataFrame,
        size: pd.DataFrame,
        profitability: pd.DataFrame,  
        market_cap: pd.DataFrame | None = None,
        quantiles: Tuple[int, int] = (2, 3),
        value_weighted: bool = True,
    ) -> pd.Series:
        """
        RMW：0.5×[(S,Robust−Weak) + (B,Robust−Weak)]
        """
        out = AcademicFactors._two_by_three_hl_avg(
            return_adj=stock_return,
            size_first=size,
            x_second=profitability,   
            market_cap=market_cap,
            quantiles=quantiles,
            value_weighted=value_weighted,
            dependent=True,
        ).astype(float)
        out.name = "RMW"
        return out

   
    @staticmethod
    def cma(
        stock_return: pd.DataFrame,
        size: pd.DataFrame,
        investment: pd.DataFrame,     
        market_cap: pd.DataFrame | None = None,
        quantiles: Tuple[int, int] = (2, 3),
        value_weighted: bool = True,
    ) -> pd.Series:
        """
        CMA：0.5×[(S,Conservative−Aggressive) + (B,Conservative−Aggressive)]
        """
        out = AcademicFactors._two_by_three_hl_avg(
            return_adj=stock_return,
            size_first=size,
            x_second=investment,       
            market_cap=market_cap,
            quantiles=quantiles,
            value_weighted=value_weighted,
            dependent=True,
        ).astype(float)
        out.name = "CMA"
        return out

   
    @staticmethod
    def momentum(
        stock_return: pd.DataFrame,
        size: pd.DataFrame,
        momentum_signal: pd.DataFrame,  
        market_cap: pd.DataFrame | None = None,
        quantiles: Tuple[int, int] = (2, 3),
        value_weighted: bool = True,
    ) -> pd.Series:
        """
        UMD（Carhart）：（dependent=False），
        0.5×[(S,Winner−Loser) + (B,Winner−Loser)]。
        """
        out = AcademicFactors._two_by_three_hl_avg(
            return_adj=stock_return,
            size_first=size,
            x_second=momentum_signal, 
            market_cap=market_cap,
            quantiles=quantiles,
            value_weighted=value_weighted,
            dependent=False,          
        ).astype(float)
        out.name = "MOM"
        return out

def bundle_capm(
    stock_return: pd.DataFrame,
    market_cap: pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
) -> List[pd.Series]:
    
    mkt = AcademicFactors.market_excess(stock_return, market_cap, risk_free_rate)
    return [mkt]


def bundle_ff3(
    stock_return: pd.DataFrame,
    size: pd.DataFrame,
    book_to_market: pd.DataFrame,
    market_cap: pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
) -> List[pd.Series]:
    
    mkt = AcademicFactors.market_excess(stock_return, market_cap, risk_free_rate)
    smb = AcademicFactors.smb_ff3(stock_return, size, book_to_market, market_cap)
    hml = AcademicFactors.hml(stock_return, size, book_to_market, market_cap)
    return [mkt, smb, hml]


def bundle_ff5(
    stock_return: pd.DataFrame,
    size: pd.DataFrame,
    book_to_market: pd.DataFrame,
    profitability: pd.DataFrame,
    investment: pd.DataFrame,       
    market_cap: pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
) -> List[pd.Series]:
   
    mkt = AcademicFactors.market_excess(stock_return, market_cap, risk_free_rate)
    smb = AcademicFactors.smb_ff5(stock_return, size, book_to_market, profitability, investment, market_cap)
    hml = AcademicFactors.hml(stock_return, size, book_to_market, market_cap)
    rmw = AcademicFactors.rmw(stock_return, size, profitability, market_cap)
    cma = AcademicFactors.cma(stock_return, size, investment, market_cap)
    return [mkt, smb, hml, rmw, cma]


def bundle_ff3_mom(
    stock_return: pd.DataFrame,
    size: pd.DataFrame,
    book_to_market: pd.DataFrame,
    momentum_signal: pd.DataFrame,  
    market_cap: pd.DataFrame,
    risk_free_rate: pd.Series | pd.DataFrame,
) -> List[pd.Series]:
   
    mkt, smb, hml = bundle_ff3(stock_return, size, book_to_market, market_cap, risk_free_rate)
    mom = AcademicFactors.momentum(stock_return, size, momentum_signal, market_cap)
    return [mkt, smb, hml, mom]
