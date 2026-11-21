"""Explicit costs and cash already included in the upstream NAV."""
from dataclasses import dataclass
import math
import pandas as pd
COST_FIELDS = ('option_commission_rate', 'future_commission_rate', 'option_slippage_ticks', 'future_slippage_points')

@dataclass
class CostBook:
    table: pd.DataFrame

    @classmethod
    def from_config(cls, cfg):
        if cfg.get('cost_file'):
            t = pd.read_csv(cfg['cost_file'], parse_dates=['effective_date'])
        else:
            missing = [k for k in COST_FIELDS if cfg.get(k) is None]
            if missing or not cfg.get('cost_source'):
                raise ValueError('미확정 파생 거래비용: ' + ', '.join(missing + ([] if cfg.get('cost_source') else ['cost_source'])) + '. risk.cost_file 또는 명시적인 비용을 입력하세요. 0도 명시적으로 입력해야 합니다.')
            t = pd.DataFrame([{**{k: cfg[k] for k in COST_FIELDS}, 'effective_date': pd.Timestamp('1900-01-01'), 'source': cfg['cost_source']}])
        if not {'effective_date', 'source', *COST_FIELDS}.issubset(t) or t.empty:
            raise ValueError('비용표에 적용일·출처·네 가지 비용 열과 최소 한 행이 필요합니다.')
        t['effective_date'] = pd.to_datetime(t.effective_date)
        if t.effective_date.isna().any() or t.effective_date.duplicated().any():
            raise ValueError('Invalid cost dates')
        for k in COST_FIELDS:
            t[k] = pd.to_numeric(t[k], errors='raise')
            if not t[k].map(lambda x: math.isfinite(x) and x >= 0).all():
                raise ValueError(f'Invalid cost {k}')
        if t.source.isna().any() or t.source.astype(str).str.strip().eq('').any():
            raise ValueError('비용의 출처 또는 팀 적용기준이 필요합니다.')
        return cls(t.sort_values('effective_date').reset_index(drop=True))

    def at(self, date):
        t = self.table[self.table.effective_date <= date]
        if t.empty:
            raise ValueError(f'{date.date()} 이전 적용 거래비용이 없습니다.')
        return t.iloc[-1]

    def fee(self, kind, qty, price, mult, date):
        c = self.at(date)
        if kind == 'PUT':
            return abs(qty) * mult * (price * c.option_commission_rate + (0.01 if price < 10 else 0.05) * c.option_slippage_ticks)
        return abs(qty) * mult * (price * c.future_commission_rate + c.future_slippage_points)

def attach_cash_limits(features, cash_file=None):
    f = features.copy()
    if cash_file:
        c = pd.read_csv(cash_file, parse_dates=['date'])
        required = ['date', 'strategy_id', 'hedge_cash_available']
        if not set(required).issubset(c) or c.duplicated(['date', 'strategy_id']).any():
            raise ValueError('현금 파일: date,strategy_id,hedge_cash_available 열과 고유 키가 필요합니다.')
        f = f.drop(columns=['hedge_cash_available'], errors='ignore').merge(c[required], on=['date', 'strategy_id'], how='left', validate='one_to_one')
    elif 'hedge_cash_available' not in f:
        f['hedge_cash_available'] = f.base_cash
    for k in ['base_nav', 'base_cash', 'hedge_cash_available']:
        if not pd.to_numeric(f[k], errors='coerce').map(math.isfinite).all():
            raise ValueError(f'Missing or nonfinite {k}')
    if (f.base_cash < -1e-05).any() or (f.base_cash > f.base_nav + 1e-05).any():
        raise ValueError('기초 NAV에 포함된 현금 잔액 오류')
    if (f.hedge_cash_available < 0).any() or (f.hedge_cash_available > f.base_cash + 1e-05).any():
        raise ValueError('헤지 현금은 기초 NAV에 이미 포함된 현금 이하여야 합니다. 별도 자금으로 더하지 않습니다.')
    return f
