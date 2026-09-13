









import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import lightgbm as lgb
from pathlib import Path
import time
import warnings
import gc

warnings.filterwarnings('ignore')




DATA_DIR = Path(r"D:\旅游流动\旅游流结果")
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(exist_ok=True)


FILE_CITY_DIM = DATA_DIR / "城市维表.csv"
FILE_TOURISM_RAW = DATA_DIR / "国内旅游人数.xlsx"
FILE_POI = DATA_DIR / "city_year_tourism_poi_counts.csv"
FILE_EDGES = DATA_DIR / "Edges_City_By_Departure_Date.csv"
FILE_CALENDAR = DATA_DIR / "逐日日历_2012-2021.csv"
FILE_DISTANCE = DATA_DIR / "city_distance_matrix_339.csv"
FILE_CITY_PANEL = DATA_DIR / "中国城市数据面板数据（2000-2024年）.xlsx"


CLIMATE_DIR = DATA_DIR

YEARS = list(range(2012, 2022))



SKIP_PREDICT_IF_EXISTS = True

print("=" * 70)
print("城市级逐日旅游OD流量预测 — 完整Pipeline")
print(f"数据目录: {DATA_DIR}")
print(f"年份范围: {YEARS[0]}-{YEARS[-1]}")
print("=" * 70)





print("\n\n")
print("╔" + "═" * 68 + "╗")
print("║          STEP 1: 年度旅游人数补全 (Log-linear OLS)           ║")
print("╚" + "═" * 68 + "╝")
print("""
模型: ln(Tourist_it) = β·ln(POI_it) + δ_{province(i), t} + ε
  - δ_{province, year} 吸收省级年度共同变化
  - β 刻画POI资源禀赋对旅游人数的边际贡献
  - Duan (1983) smearing estimator 偏差修正
""")

t0 = time.time()

OUTLIER_THRESHOLD = np.log(10)


print("[Step1] 1/6 读取数据...")
dim = pd.read_csv(FILE_CITY_DIM, encoding="utf-8-sig")
dim = dim[["city_name", "city_code_ref"]].copy()
dim["city_name"] = dim["city_name"].str.strip()

tourism_wide = pd.read_excel(FILE_TOURISM_RAW, sheet_name="Sheet2")
tourism_wide["城市"] = tourism_wide["城市"].str.strip()
tourism_wide["省份"] = tourism_wide["省份"].str.strip()

year_cols = {f"{y}年": y for y in YEARS}
tourism_long = tourism_wide.melt(
    id_vars=["省份", "城市"],
    value_vars=list(year_cols.keys()),
    var_name="year_str",
    value_name="tourists_10k",
)
tourism_long["year"] = tourism_long["year_str"].map(year_cols)
tourism_long = tourism_long.dropna(subset=["year"])
tourism_long = tourism_long[["省份", "城市", "year", "tourists_10k"]].rename(
    columns={"城市": "city_name", "省份": "province"}
)

poi = pd.read_csv(FILE_POI, encoding="utf-8-sig")
poi["city"] = poi["city"].str.strip()
poi["province"] = poi["province"].str.strip()
poi = poi[poi["year"].isin(YEARS)].copy()
print(f"  城市维表: {len(dim)} 城市")
print(f"  旅游人数: {len(tourism_long)} 条")
print(f"  POI数据: {len(poi)} 条")


print("[Step1] 2/6 异常值清洗...")

def detect_extreme_outliers(group):

    vals = group["tourists_10k"].dropna()
    if len(vals) < 3:
        group["is_outlier"] = False
        return group
    log_median = np.log(vals).median()
    group["is_outlier"] = False
    mask = group["tourists_10k"].notna()
    log_all = np.log(group.loc[mask, "tourists_10k"])
    group.loc[mask, "is_outlier"] = np.abs(log_all - log_median) > OUTLIER_THRESHOLD
    return group

tourism_long = tourism_long.groupby("city_name", group_keys=False).apply(
    detect_extreme_outliers
)
n_outliers = tourism_long["is_outlier"].sum()
print(f"  剔除极端异常值: {n_outliers} 条")
tourism_long.loc[tourism_long["is_outlier"], "tourists_10k"] = np.nan


print("[Step1] 3/6 构建面板...")

province_map_step1 = (
    poi[poi["city"].isin(dim["city_name"])]
    .groupby("city")["province"].first().to_dict()
)

panel_step1 = pd.MultiIndex.from_product(
    [dim["city_name"], YEARS], names=["city_name", "year"]
).to_frame(index=False)
panel_step1["province"] = panel_step1["city_name"].map(province_map_step1)

poi_subset = poi[["city", "year", "uniq10_n"]].rename(
    columns={"city": "city_name", "uniq10_n": "poi_total"}
)
panel_step1 = panel_step1.merge(poi_subset, on=["city_name", "year"], how="left")
panel_step1 = panel_step1.merge(
    tourism_long[["city_name", "year", "tourists_10k"]],
    on=["city_name", "year"], how="left",
)

poi_city_mean = panel_step1.groupby("city_name")["poi_total"].transform("mean")
panel_step1["poi_total"] = panel_step1["poi_total"].fillna(poi_city_mean)
panel_step1["ln_poi"] = np.log(panel_step1["poi_total"])
panel_step1["ln_tourists"] = np.log(panel_step1["tourists_10k"])
panel_step1["has_obs"] = panel_step1["tourists_10k"].notna()
panel_step1["prov_year"] = panel_step1["province"] + "_" + panel_step1["year"].astype(str)

print(f"  面板: {len(panel_step1)} 行 | 有观测: {panel_step1['has_obs'].sum()} | "
      f"需预测: {(~panel_step1['has_obs']).sum()}")


print("[Step1] 4/6 Log-linear OLS...")

train_step1 = panel_step1[panel_step1["has_obs"]].copy()
train_prov_years = set(train_step1["prov_year"].unique())

model_step1 = smf.ols("ln_tourists ~ ln_poi + C(prov_year)", data=train_step1).fit()

print(f"  R² = {model_step1.rsquared:.4f} | Adj R² = {model_step1.rsquared_adj:.4f} | "
      f"N = {model_step1.nobs:.0f}")
print(f"  POI弹性 β = {model_step1.params['ln_poi']:.4f} "
      f"(p = {model_step1.pvalues['ln_poi']:.2e})")


print("[Step1] 5/6 预测缺失值...")

mask_known = panel_step1["prov_year"].isin(train_prov_years)
panel_step1.loc[mask_known, "ln_pred"] = model_step1.predict(panel_step1[mask_known])

unseen_mask = ~mask_known
n_unseen = unseen_mask.sum()
if n_unseen > 0:
    unseen_rows = panel_step1[unseen_mask].copy()
    print(f"  未见省×年组合: {n_unseen} 行")

    train_prov_year_map = train_step1.groupby("province")["year"].apply(set).to_dict()

    def find_nearest_prov_year(row):
        prov = row["province"]
        yr = row["year"]
        if prov not in train_prov_year_map:
            return None
        available = sorted(train_prov_year_map[prov])
        nearest = min(available, key=lambda y: abs(y - yr))
        return f"{prov}_{nearest}"

    unseen_rows["prov_year"] = unseen_rows.apply(find_nearest_prov_year, axis=1)

    no_proxy = unseen_rows["prov_year"].isna()
    if no_proxy.any():
        ref_prov_year = sorted(train_prov_years)[0]
        unseen_rows.loc[no_proxy, "prov_year"] = ref_prov_year

    panel_step1.loc[unseen_mask, "ln_pred"] = model_step1.predict(unseen_rows)

panel_step1["pred_10k"] = np.exp(panel_step1["ln_pred"])

smearing_factor = np.exp(model_step1.resid).mean()
panel_step1["pred_adj_10k"] = panel_step1["pred_10k"] * smearing_factor
print(f"  Smearing factor: {smearing_factor:.4f}")

panel_step1["tourists_final_10k"] = np.where(
    panel_step1["has_obs"], panel_step1["tourists_10k"], panel_step1["pred_adj_10k"]
)


print("[Step1] 6/6 评估与输出...")

both = panel_step1[panel_step1["has_obs"]].copy()
both["ratio"] = both["pred_adj_10k"] / both["tourists_10k"]
print(f"  中位 ratio: {both['ratio'].median():.3f}")
print(f"  ratio ∈ [0.5, 2]: {((both['ratio']>=0.5)&(both['ratio']<=2)).mean()*100:.1f}%")
print(f"  MAPE: {(both['ratio']-1).abs().mean()*100:.1f}%")

output_step1 = panel_step1[[
    "city_name", "year", "province", "poi_total",
    "tourists_10k", "pred_adj_10k", "tourists_final_10k", "has_obs"
]].copy()
output_step1 = output_step1.rename(columns={
    "tourists_10k": "tourists_official_10k",
    "pred_adj_10k": "tourists_model_10k",
})
output_step1 = output_step1.sort_values(["city_name", "year"]).reset_index(drop=True)

step1_path = OUTPUT_DIR / "annual_tourism_panel_339x10.csv"
output_step1.to_csv(step1_path, index=False, encoding="utf-8-sig")
print(f"  输出: {step1_path}")
print(f"  {output_step1['city_name'].nunique()} 城市 × {len(YEARS)} 年 = {len(output_step1)} 行")


summary_path = OUTPUT_DIR / "ols_model_summary.txt"
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(str(model_step1.summary()))
    f.write(f"\n\nSmearing factor: {smearing_factor:.4f}")
    f.write(f"\nOutliers removed: {n_outliers}")

print(f"\n  Step 1 完成, 耗时 {time.time()-t0:.1f}s")


del train_step1, tourism_long, tourism_wide, both
gc.collect()






print("\n\n")
print("╔" + "═" * 68 + "╗")
print("║   STEP 3: 逐日旅游人数分配 (LightGBM + 环境/日历/城市特征)   ║")
print("╚" + "═" * 68 + "╝")
print("""
模型: LightGBM梯度提升决策树
      目标: 城市级逐日旅游出行量 (ln(1+y))
      特征: 气候 + 日历结构 + 城市属性 + 城市月度季节形态(仅训练集城市)
      缺失: 气候按城市-月气候态填补, 其余保留NaN交LightGBM原生处理
      约束: Tourist_daily(i,d) = Annual(i,y) × pred(i,d) / Σ_d pred(i,d')
""")

t1 = time.time()




print("[Step3] PART 1: 加载基础数据...")


panel_s3 = pd.read_csv(step1_path)
panel_cities = sorted(panel_s3["city_name"].unique())
province_map = panel_s3.groupby("city_name")["province"].first().to_dict()
print(f"  年度面板: {len(panel_cities)}城市")


edges = pd.read_csv(FILE_EDGES,
                    usecols=["Flow_Date", "Target_City_ZH", "Trip_Count"])
edges['date'] = pd.to_datetime(edges['Flow_Date'])
edges = edges[(edges.date.dt.year >= 2012) & (edges.date.dt.year <= 2021)]

city_daily_trips = (edges.groupby(['Target_City_ZH', 'date'])['Trip_Count']
                    .sum().reset_index())
city_daily_trips.columns = ['city_name', 'date', 'trips']
city_daily_trips['year'] = city_daily_trips['date'].dt.year
print(f"  城市日度训练样本: {len(city_daily_trips):,}条, "
      f"{city_daily_trips.city_name.nunique()}城市")




all_train_cities = city_daily_trips['city_name'].unique().copy()
np.random.seed(42)
np.random.shuffle(all_train_cities)
n_train_cities = int(len(all_train_cities) * 0.8)
train_city_set = set(all_train_cities[:n_train_cities])
val_city_set = set(all_train_cities[n_train_cities:])
print(f"  城市划分: 训练{len(train_city_set)}城市 / 验证{len(val_city_set)}城市")


print("  计算城市月度季节形态 (仅训练集城市)...")
edges_seasonal = edges[edges['Target_City_ZH'].isin(train_city_set)].copy()
edges_seasonal['month'] = edges_seasonal['date'].dt.month

city_month_agg = (edges_seasonal.groupby(['Target_City_ZH', 'month'])['Trip_Count']
                  .sum().reset_index())
city_month_agg.columns = ['city_name', 'month', 'month_trips']

city_total_diary = city_month_agg.groupby('city_name')['month_trips'].sum().reset_index()
city_total_diary.columns = ['city_name', 'total_trips']
city_month_agg = city_month_agg.merge(city_total_diary, on='city_name')

city_month_agg['month_ratio'] = city_month_agg['month_trips'] / city_month_agg['total_trips']


seasonal_wide = city_month_agg.pivot(index='city_name', columns='month',
                                      values='month_ratio').fillna(0)
seasonal_wide.columns = [f'seasonal_m{int(m):02d}' for m in seasonal_wide.columns]
seasonal_wide = seasonal_wide.reset_index()


all_339_cities = list(province_map.keys())
seasonal_wide = (pd.DataFrame({'city_name': all_339_cities})
                 .merge(seasonal_wide, on='city_name', how='left'))
seasonal_cols_12 = [f'seasonal_m{m:02d}' for m in range(1, 13)]
for col in seasonal_cols_12:
    if col not in seasonal_wide.columns:
        seasonal_wide[col] = np.nan


city_total_map = city_total_diary.set_index('city_name')['total_trips'].to_dict()
fallback_cities_set = set(c for c in all_339_cities
                          if c not in train_city_set
                          or city_total_map.get(c, 0) < 30)
n_val_fallback = len(fallback_cities_set & val_city_set)

seasonal_wide['province'] = seasonal_wide['city_name'].map(province_map)

donor_mask = ~seasonal_wide['city_name'].isin(fallback_cities_set)
prov_seasonal_mean = (seasonal_wide[donor_mask]
                      .groupby('province')[seasonal_cols_12].mean())

natl_seasonal_mean = seasonal_wide.loc[donor_mask, seasonal_cols_12].mean()


for idx in seasonal_wide.index:
    if seasonal_wide.loc[idx, 'city_name'] in fallback_cities_set:
        prov = seasonal_wide.loc[idx, 'province']
        if prov in prov_seasonal_mean.index:
            seasonal_wide.loc[idx, seasonal_cols_12] = prov_seasonal_mean.loc[prov].values
        else:
            seasonal_wide.loc[idx, seasonal_cols_12] = natl_seasonal_mean.values

seasonal_wide[seasonal_cols_12] = seasonal_wide[seasonal_cols_12].fillna(natl_seasonal_mean)


seasonal_wide.drop(columns=['seasonal_m01', 'province'], inplace=True)
seasonal_cols_s3 = [c for c in seasonal_wide.columns if c.startswith('seasonal_m')]
print(f"  城市季节形态: {seasonal_wide.city_name.nunique()}城市, "
      f"{len(seasonal_cols_s3)}维特征, "
      f"回退城市{len(fallback_cities_set)}个 (其中验证集城市{n_val_fallback}个)")
del edges_seasonal, city_month_agg, city_total_diary


cal = pd.read_csv(FILE_CALENDAR)
cal["flow_date"] = pd.to_datetime(cal["flow_date"])
cal_features_s3 = ['weekday_num', 'flow_month', 'is_weekend', 'is_adjusted_workday',
                   'is_official_holiday_period', 'holiday_length', 'holiday_day_index',
                   'is_golden_week', 'rest_streak_length', 'rest_streak_position',
                   'preholiday_1d', 'preholiday_3d', 'postholiday_1d', 'postholiday_3d',
                   'days_to_next_holiday_start']
for col in cal_features_s3:
    if col in cal.columns:
        cal[col] = cal[col].fillna(0)
print(f"  日历: {len(cal)}天")


climate_frames_s3 = []
for year in YEARS:
    fpath = CLIMATE_DIR / f"城市逐日气候_{year}.csv"
    if fpath.exists():
        clim = pd.read_csv(fpath)
        clim['flow_date'] = pd.to_datetime(clim['flow_date'])
        climate_frames_s3.append(clim[['flow_date', 'city_name', 'tmean_c',
                                       'precip_mm']])
climate_all_s3 = pd.concat(climate_frames_s3, ignore_index=True)
print(f"  气候数据: {len(climate_all_s3):,}条")





_clim_tmp = climate_all_s3[['city_name', 'flow_date', 'tmean_c', 'precip_mm']].copy()
_clim_tmp['month'] = _clim_tmp['flow_date'].dt.month
climate_norm = (_clim_tmp.groupby(['city_name', 'month'])[['tmean_c', 'precip_mm']]
                .mean().reset_index()
                .rename(columns={'tmean_c': 'tmean_clim',
                                 'precip_mm': 'precip_clim'}))
print(f"  气候态表: {climate_norm.city_name.nunique()}城市 × 12月 "
      f"({len(climate_norm)}条)")
del _clim_tmp


poi_df_s3 = pd.read_csv(FILE_POI)
poi_cols_s3 = ['scenic_n', 'museum_n', 'lodging_n', 'resort_n', 'uniq10_n']
print(f"  POI数据: {poi_df_s3.shape[0]}条")


city_panel_s3 = pd.read_excel(FILE_CITY_PANEL, sheet_name='线性插值')
panel_features_raw_s3 = {
    '地区生产总值(万元)': 'gdp',
    '第三产业增加值(万元)': 'tertiary_industry',
    '户籍人口(万人)': 'huji_pop',
    '人均地区生产总值(元)': 'gdp_per_capita'
}
panel_cols_keep_s3 = ['年份', '城市'] + list(panel_features_raw_s3.keys())
city_panel_s3 = city_panel_s3[[c for c in panel_cols_keep_s3
                               if c in city_panel_s3.columns]].copy()
city_panel_s3.rename(columns={'年份': 'year', '城市': 'city_name'}, inplace=True)
city_panel_s3.rename(columns=panel_features_raw_s3, inplace=True)
panel_feats_s3 = list(panel_features_raw_s3.values())

city_panel_s3['province'] = city_panel_s3['city_name'].map(province_map)
prov_means_s3 = city_panel_s3.groupby(['year', 'province'])[panel_feats_s3].transform('mean')
for col in panel_feats_s3:
    city_panel_s3[col] = city_panel_s3[col].fillna(prov_means_s3[col])
print(f"  城市面板: {city_panel_s3.city_name.nunique()}城市")





print("\n[Step3] PART 2: 特征工程...")



cal_features_zero_fill = [
    'is_weekend', 'is_adjusted_workday', 'is_official_holiday_period',
    'holiday_length', 'holiday_day_index', 'is_golden_week',
    'rest_streak_length', 'rest_streak_position',
    'preholiday_1d', 'preholiday_3d', 'postholiday_1d', 'postholiday_3d',
    'days_to_next_holiday',
]


def build_features_s3(df, climate_all_s3, cal, poi_df_s3, city_panel_s3, seasonal_wide):

    feats = pd.DataFrame(index=df.index)
    df_year = df['year'] if 'year' in df.columns else df['date'].dt.year

    
    
    clim_merge = df[['city_name', 'date']].merge(
        climate_all_s3.rename(columns={'flow_date': 'date'}),
        on=['city_name', 'date'], how='left')
    clim_merge['month'] = clim_merge['date'].dt.month
    clim_merge = clim_merge.merge(climate_norm, on=['city_name', 'month'], how='left')
    feats['tmean_c'] = clim_merge['tmean_c'].fillna(clim_merge['tmean_clim']).values
    feats['precip_mm'] = clim_merge['precip_mm'].fillna(clim_merge['precip_clim']).values

    
    cal_merge = df[['date']].merge(
        cal.rename(columns={'flow_date': 'date'}),
        on='date', how='left')
    feats['day_of_week'] = cal_merge['weekday_num'].values
    feats['month'] = cal_merge['flow_month'].values
    feats['is_weekend'] = cal_merge['is_weekend'].values
    feats['is_adjusted_workday'] = cal_merge['is_adjusted_workday'].values
    feats['is_official_holiday_period'] = cal_merge['is_official_holiday_period'].values
    feats['holiday_length'] = cal_merge['holiday_length'].values
    feats['holiday_day_index'] = cal_merge['holiday_day_index'].values
    feats['is_golden_week'] = cal_merge['is_golden_week'].values
    feats['rest_streak_length'] = cal_merge['rest_streak_length'].values
    feats['rest_streak_position'] = cal_merge['rest_streak_position'].values
    feats['preholiday_1d'] = cal_merge['preholiday_1d'].values
    feats['preholiday_3d'] = cal_merge['preholiday_3d'].values
    feats['postholiday_1d'] = cal_merge['postholiday_1d'].values
    feats['postholiday_3d'] = cal_merge['postholiday_3d'].values
    feats['days_to_next_holiday'] = cal_merge['days_to_next_holiday_start'].values

    
    poi_merge = df[['city_name']].copy()
    poi_merge['year'] = df_year.values
    poi_merge = poi_merge.merge(
        poi_df_s3.rename(columns={'city': 'city_name'}),
        on=['city_name', 'year'], how='left')
    for col in poi_cols_s3:
        feats[col] = poi_merge[col].values if col in poi_merge.columns else np.nan

    panel_merge = df[['city_name']].copy()
    panel_merge['year'] = df_year.values
    panel_merge = panel_merge.merge(
        city_panel_s3[['city_name', 'year'] + panel_feats_s3],
        on=['city_name', 'year'], how='left')
    for col in panel_feats_s3:
        feats[col] = panel_merge[col].values if col in panel_merge.columns else np.nan

    
    season_merge = df[['city_name']].merge(
        seasonal_wide, on='city_name', how='left')
    for col in seasonal_cols_s3:
        feats[col] = season_merge[col].values if col in season_merge.columns else np.nan

    
    
    
    
    
    feats[cal_features_zero_fill] = feats[cal_features_zero_fill].fillna(0)
    return feats



print("  构建训练集特征...")
city_daily_trips = city_daily_trips.merge(
    cal[['flow_date', 'flow_month']].rename(columns={'flow_date': 'date'}),
    on='date', how='left')

train_features_s3 = build_features_s3(city_daily_trips, climate_all_s3,
                                       cal, poi_df_s3, city_panel_s3, seasonal_wide)
feature_cols_s3 = list(train_features_s3.columns)
print(f"  特征维度: {len(feature_cols_s3)}")
print(f"  特征列表: {feature_cols_s3}")





print("\n[Step3] PART 3: 训练LightGBM模型...")

X_s3 = train_features_s3.values.astype(np.float32)
y_s3 = city_daily_trips['trips'].values.astype(np.float32)
y_s3_log = np.log1p(y_s3)


train_mask_s3 = city_daily_trips['city_name'].isin(train_city_set)
val_mask_s3 = ~train_mask_s3

X_train_s3 = X_s3[train_mask_s3.values]
X_val_s3 = X_s3[val_mask_s3.values]
y_train_s3 = y_s3_log[train_mask_s3.values]
y_val_s3 = y_s3_log[val_mask_s3.values]
print(f"  训练集: {len(X_train_s3):,}条 ({n_train_cities}城市)")
print(f"  验证集: {len(X_val_s3):,}条 ({len(all_train_cities)-n_train_cities}城市)")

params_s3 = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 30,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1
}

dtrain_s3 = lgb.Dataset(X_train_s3, label=y_train_s3, feature_name=feature_cols_s3)
dval_s3 = lgb.Dataset(X_val_s3, label=y_val_s3, feature_name=feature_cols_s3,
                      reference=dtrain_s3)

lgb_model_s3 = lgb.train(
    params_s3, dtrain_s3,
    num_boost_round=2000,
    valid_sets=[dtrain_s3, dval_s3],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(200)
    ]
)

print(f"\n  最佳迭代: {lgb_model_s3.best_iteration}")
print(f"  验证RMSE: {lgb_model_s3.best_score['valid_1']['rmse']:.4f}")


step3_metrics = {
    'best_iteration': lgb_model_s3.best_iteration,
    'max_rounds': 2000,
    'train_rmse': lgb_model_s3.best_score['training']['rmse'],
    'val_rmse': lgb_model_s3.best_score['valid_1']['rmse'],
    'n_train': len(X_train_s3),
    'n_val': len(X_val_s3),
    'n_features': len(feature_cols_s3),
    'note': 'RMSE on log1p-transformed target'
}
pd.DataFrame([step3_metrics]).to_csv(
    OUTPUT_DIR / "step3_model_metrics.csv", index=False, encoding='utf-8-sig')


importance_s3 = pd.DataFrame({
    'feature': feature_cols_s3,
    'importance': lgb_model_s3.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("  Top-10 特征重要性 (gain):")
for _, row in importance_s3.head(10).iterrows():
    print(f"    {row['feature']:<25s} {row['importance']:.0f}")
importance_s3.to_csv(OUTPUT_DIR / "step3_feature_importance.csv",
                     index=False, encoding='utf-8-sig')





print("\n[Step3] PART 4: 全量预测并约束归一化...")


results_s3 = []

for year in YEARS:
    cal_year = cal[cal["flow_year"] == year].copy()
    dates_year = cal_year["flow_date"].values
    n_days = len(dates_year)

    panel_year = panel_s3[panel_s3["year"] == year][["city_name", "tourists_final_10k"]].copy()
    panel_year = panel_year[panel_year["tourists_final_10k"].notna() &
                            (panel_year["tourists_final_10k"] > 0)]
    year_cities = panel_year["city_name"].tolist()

    if len(year_cities) == 0:
        continue

    
    pred_df = pd.DataFrame({
        'city_name': np.repeat(year_cities, n_days),
        'date': np.tile(dates_year, len(year_cities)),
    })
    pred_df['year'] = year

    
    pred_feats = build_features_s3(pred_df, climate_all_s3, cal, poi_df_s3, city_panel_s3, seasonal_wide)
    X_pred = pred_feats.values.astype(np.float32)

    
    y_pred_log = lgb_model_s3.predict(X_pred, num_iteration=lgb_model_s3.best_iteration)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0.0)

    
    
    
    n_cities_year = len(year_cities)
    annual_totals = (panel_year.set_index('city_name')['tourists_final_10k']
                     .reindex(year_cities).to_numpy(dtype=np.float64))
    pred_mat = y_pred.reshape(n_cities_year, n_days)
    row_sums = pred_mat.sum(axis=1)

    ok = row_sums > 0
    ratio = np.zeros(n_cities_year, dtype=np.float64)
    ratio[ok] = annual_totals[ok] / row_sums[ok]
    
    scaled = np.where(ok[:, None],
                      pred_mat * ratio[:, None],
                      (annual_totals / n_days)[:, None])

    results_s3.append(pd.DataFrame({
        'city_name': np.repeat(np.asarray(year_cities, dtype=object), n_days),
        'date': np.tile(dates_year, n_cities_year),
        'tourists_10k': scaled.ravel(),
    }))
    print(f"    {year}: {n_cities_year}城市 × {n_days}天")

    del pred_df, pred_feats, X_pred, y_pred_log, y_pred, pred_mat, scaled
    gc.collect()





print("\n[Step3] PART 5: 合并输出")

daily_panel = pd.concat(results_s3, ignore_index=True)
daily_panel["date"] = pd.to_datetime(daily_panel["date"])
daily_panel["year"] = daily_panel["date"].dt.year
daily_panel["month"] = daily_panel["date"].dt.month
daily_panel["day"] = daily_panel["date"].dt.day


cal_merge_s3 = cal[["flow_date", "day_status", "holiday_name",
                    "is_golden_week", "rest_streak_length",
                    "rest_streak_position"]].rename(columns={"flow_date": "date"})
daily_panel = daily_panel.merge(cal_merge_s3, on="date", how="left")
daily_panel["province"] = daily_panel["city_name"].map(province_map)

daily_panel = daily_panel[[
    "city_name", "province", "date", "year", "month", "day",
    "tourists_10k", "day_status", "holiday_name",
    "is_golden_week", "rest_streak_length", "rest_streak_position"
]].sort_values(["city_name", "date"]).reset_index(drop=True)

print(f"  最终面板: {len(daily_panel):,} 行")
print(f"  城市数: {daily_panel.city_name.nunique()}")


agg = daily_panel.groupby(["city_name", "year"])["tourists_10k"].sum().reset_index()
agg = agg.merge(panel_s3[["city_name", "year", "tourists_final_10k"]],
                on=["city_name", "year"], how="left")
agg["ratio"] = agg["tourists_10k"] / agg["tourists_final_10k"]
print(f"  年度一致性 ratio: mean={agg.ratio.mean():.6f}, "
      f"min={agg.ratio.min():.6f}, max={agg.ratio.max():.6f}")

step3_path = OUTPUT_DIR / "daily_tourism_panel_339_v2.csv"
daily_panel.to_csv(step3_path, index=False, encoding="utf-8-sig")
print(f"  输出: {step3_path}")
print(f"  Step 3 完成, 耗时 {time.time()-t1:.1f}s")


del results_s3, city_daily_trips, climate_all_s3, climate_frames_s3
del train_features_s3, X_s3, y_s3, y_s3_log, edges
del poi_df_s3, city_panel_s3, daily_panel
gc.collect()







print("\n\n")
print("╔" + "═" * 68 + "╗")
print("║     STEP 4: OD流预测 (LightGBM + 推-拉-阻力特征体系)       ║")
print("╚" + "═" * 68 + "╝")
print("""
特征体系 (5组, 29特征):
  [1] 空间阻力: distance_km
  [2] 目的地拉力: POI + 经济 + 年度游客 + 气候(tmean, precip)
  [3] 出发地推力: 人口 + 消费力 + 气候(tmean, precip_heavy)
  [4] 时间调节: 月份 + 星期 + 日历特征
  [5] 气候交互: temp_diff
""")

t2 = time.time()


print("[Step4] 1/8 加载基础数据...")

dist_df = pd.read_csv(FILE_DISTANCE, index_col=0)
cities_339 = list(dist_df.index)
city_to_idx = {c: i for i, c in enumerate(cities_339)}
dist_matrix = dist_df.values
print(f"  距离矩阵: {dist_matrix.shape}")

poi_df = pd.read_csv(FILE_POI)
poi_features = ['scenic_n', 'museum_n', 'memorial_n', 'relig_n', 'zoo_bot_n',
                'lodging_n', 'agency_n', 'resort_n', 'ski_n', 'hub_n', 'uniq10_n']
print(f"  POI数据: {poi_df.shape[0]}条")

calendar_df = pd.read_csv(FILE_CALENDAR)
calendar_df['flow_date'] = pd.to_datetime(calendar_df['flow_date'])
cal_features_s4 = ['is_weekend', 'is_adjusted_workday',
                   'is_official_holiday_period', 'holiday_length', 'holiday_day_index',
                   'is_golden_week', 'rest_streak_length',
                   'rest_streak_position',
                   'preholiday_1d', 'preholiday_3d', 'postholiday_1d', 'postholiday_3d']
for col in cal_features_s4:
    if col in calendar_df.columns:
        calendar_df[col] = calendar_df[col].fillna(0)
print(f"  日历: {len(calendar_df)}天")

annual_df = pd.read_csv(step1_path)
print(f"  年度面板: {annual_df.shape}")


print("  加载城市面板数据...")
city_panel = pd.read_excel(FILE_CITY_PANEL, sheet_name='线性插值')
panel_features_raw = {
    '地区生产总值(万元)': 'gdp',
    '第三产业增加值(万元)': 'tertiary_industry',
    '户籍人口(万人)': 'huji_pop',
    '人均地区生产总值(元)': 'gdp_per_capita'
}
panel_cols_keep = ['年份', '省份', '城市'] + list(panel_features_raw.keys())
city_panel = city_panel[[c for c in panel_cols_keep if c in city_panel.columns]].copy()
city_panel.rename(columns={'年份': 'year', '省份': 'province', '城市': 'city_name'}, inplace=True)
city_panel.rename(columns=panel_features_raw, inplace=True)
panel_features = list(panel_features_raw.values())

province_means = city_panel.groupby(['year', 'province'])[panel_features].transform('mean')
for col in panel_features:
    city_panel[col] = city_panel[col].fillna(province_means[col])
print(f"  城市面板: {city_panel.city_name.nunique()}城市")


print("\n[Step4] 2/8 计算气候特征...")

climate_all = []
for year in YEARS:
    fpath = CLIMATE_DIR / f"城市逐日气候_{year}.csv"
    if fpath.exists():
        clim = pd.read_csv(fpath)
        clim['flow_date'] = pd.to_datetime(clim['flow_date'])
        climate_all.append(clim[['flow_date', 'city_name', 'tmean_c', 'precip_mm']])
climate_all = pd.concat(climate_all, ignore_index=True)


climate_all['month'] = climate_all['flow_date'].dt.month
climate_norm_s4 = (climate_all.groupby(['city_name', 'month'])[['tmean_c', 'precip_mm']]
                   .mean().reset_index()
                   .rename(columns={'tmean_c': 'tmean_clim',
                                    'precip_mm': 'precip_clim'}))
climate_all = climate_all.merge(climate_norm_s4, on=['city_name', 'month'], how='left')
climate_all['tmean_c'] = climate_all['tmean_c'].fillna(climate_all['tmean_clim'])
climate_all['precip_mm'] = climate_all['precip_mm'].fillna(climate_all['precip_clim'])


climate_all['precip_heavy'] = np.where(
    climate_all['precip_mm'].notna(),
    (climate_all['precip_mm'] > 10).astype(float),
    np.nan)
climate_all = climate_all[['flow_date', 'city_name', 'tmean_c', 'precip_mm',
                           'precip_heavy']]
del climate_norm_s4
print(f"  气候数据: {len(climate_all)}条")


print("\n[Step4] 3/8 准备训练数据...")

edges_s4 = pd.read_csv(FILE_EDGES)
edges_s4['Flow_Date'] = pd.to_datetime(edges_s4['Flow_Date'])
edges_s4 = edges_s4[(edges_s4['Flow_Date'].dt.year >= 2012) & (edges_s4['Flow_Date'].dt.year <= 2021)]
edges_s4 = edges_s4[edges_s4['Source_City_ZH'].isin(cities_339) & edges_s4['Target_City_ZH'].isin(cities_339)]

train_pos = edges_s4.groupby(['Source_City_ZH', 'Target_City_ZH', 'Flow_Date']).agg(
    trip_count=('Trip_Count', 'sum'),
    trip_weight=('Trip_Normalized_Weight', 'sum')
).reset_index()
train_pos.rename(columns={'Source_City_ZH': 'origin', 'Target_City_ZH': 'dest',
                          'Flow_Date': 'date'}, inplace=True)
train_pos['label'] = train_pos['trip_weight']
train_pos['is_positive'] = 1
print(f"  正样本: {len(train_pos)}条")


print("  负采样中...")
np.random.seed(42)
NEG_RATIO = 5

neg_samples = []
grouped = train_pos.groupby(['origin', 'date'])['dest'].apply(set).to_dict()
for (origin, date), pos_dests in grouped.items():
    candidates = [c for c in cities_339 if c != origin and c not in pos_dests]
    n_neg = min(NEG_RATIO * len(pos_dests), len(candidates))
    if n_neg > 0:
        neg_dests = np.random.choice(candidates, size=n_neg, replace=False)
        for d in neg_dests:
            neg_samples.append({'origin': origin, 'dest': d, 'date': date,
                              'trip_count': 0, 'trip_weight': 0,
                              'label': 0, 'is_positive': 0})

train_neg = pd.DataFrame(neg_samples)
print(f"  负样本: {len(train_neg)}条")

train_data = pd.concat([train_pos, train_neg], ignore_index=True)
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
print(f"  总训练集: {len(train_data)}条 (正负比 1:{NEG_RATIO})")
del neg_samples, train_neg, edges_s4
gc.collect()


print("\n[Step4] 4/8 特征工程...")

def build_features(df, dist_matrix, city_to_idx, poi_df, city_panel,
                   climate_all, calendar_df, annual_df):

    feats = pd.DataFrame(index=df.index)
    df_year = df['year'] if 'year' in df.columns else df['date'].dt.year

    
    origin_idx = df['origin'].map(city_to_idx)
    dest_idx = df['dest'].map(city_to_idx)
    valid_mask = origin_idx.notna() & dest_idx.notna()

    distances = np.full(len(df), np.nan)
    valid_o = origin_idx[valid_mask].astype(int).values
    valid_d = dest_idx[valid_mask].astype(int).values
    distances[valid_mask.values] = dist_matrix[valid_o, valid_d]

    feats['distance_km'] = distances

    
    dest_poi = df[['dest']].copy()
    dest_poi['year'] = df_year.values
    dest_poi = dest_poi.merge(poi_df.rename(columns={'city': 'dest'}),
                              on=['dest', 'year'], how='left')
    dest_poi_cols = ['scenic_n', 'museum_n', 'lodging_n', 'resort_n']
    for col in dest_poi_cols:
        feats[f'dest_{col}'] = dest_poi[col].values if col in dest_poi.columns else np.nan

    dest_panel = df[['dest']].copy()
    dest_panel['year'] = df_year.values
    dest_panel = dest_panel.merge(
        city_panel.rename(columns={'city_name': 'dest'}),
        on=['dest', 'year'], how='left')
    for col in ['gdp', 'tertiary_industry']:
        feats[f'dest_{col}'] = dest_panel[col].values if col in dest_panel.columns else np.nan

    dest_annual = df[['dest']].copy()
    dest_annual['year'] = df_year.values
    dest_annual = dest_annual.merge(
        annual_df.rename(columns={'city_name': 'dest'})[['dest', 'year', 'tourists_final_10k']],
        on=['dest', 'year'], how='left')
    feats['dest_annual_tourists'] = dest_annual['tourists_final_10k'].values

    dest_clim = df[['dest', 'date']].merge(
        climate_all.rename(columns={'city_name': 'dest', 'flow_date': 'date'}),
        on=['dest', 'date'], how='left')
    feats['dest_tmean'] = dest_clim['tmean_c'].values
    feats['dest_precip'] = dest_clim['precip_mm'].values

    
    orig_panel = df[['origin']].copy()
    orig_panel['year'] = df_year.values
    orig_panel = orig_panel.merge(
        city_panel.rename(columns={'city_name': 'origin'}),
        on=['origin', 'year'], how='left')
    feats['orig_huji_pop'] = orig_panel['huji_pop'].values if 'huji_pop' in orig_panel.columns else np.nan
    feats['orig_gdp_per_capita'] = orig_panel['gdp_per_capita'].values if 'gdp_per_capita' in orig_panel.columns else np.nan

    orig_clim = df[['origin', 'date']].merge(
        climate_all.rename(columns={'city_name': 'origin', 'flow_date': 'date'}),
        on=['origin', 'date'], how='left')
    feats['orig_tmean'] = orig_clim['tmean_c'].values
    feats['orig_precip_heavy'] = orig_clim['precip_heavy'].values

    
    cal_merge = df[['date']].merge(
        calendar_df.rename(columns={'flow_date': 'date'}),
        on='date', how='left')
    for col in cal_features_s4:
        if col in cal_merge.columns:
            feats[col] = cal_merge[col].values

    feats['month'] = df['date'].dt.month
    feats['day_of_week'] = df['date'].dt.dayofweek

    
    feats['temp_diff'] = feats['orig_tmean'] - feats['dest_tmean']

    
    
    
    _cal_fill = [c for c in cal_features_s4 if c in feats.columns]
    if _cal_fill:
        feats[_cal_fill] = feats[_cal_fill].fillna(0)
    return feats

print("  构建训练集特征...")
train_features = build_features(train_data, dist_matrix, city_to_idx,
                                poi_df, city_panel, climate_all, calendar_df, annual_df)
feature_cols = list(train_features.columns)
print(f"  特征维度: {len(feature_cols)}")


print("\n[Step4] 5/8 训练LightGBM模型...")

X = train_features.values.astype(np.float32)
y = train_data['label'].values.astype(np.float32)
y_log = np.log1p(y)

train_data['od_pair'] = train_data['origin'] + '_' + train_data['dest']
od_pairs = train_data['od_pair'].unique()
np.random.seed(42)
np.random.shuffle(od_pairs)
n_train = int(len(od_pairs) * 0.8)
train_pairs = set(od_pairs[:n_train])
train_mask = train_data['od_pair'].isin(train_pairs)
val_mask = ~train_mask

X_train, X_val = X[train_mask.values], X[val_mask.values]
y_train, y_val = y_log[train_mask.values], y_log[val_mask.values]
print(f"  训练集: {len(X_train)}, 验证集: {len(X_val)}")

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1
}

dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain)

lgb_model = lgb.train(
    params, dtrain,
    num_boost_round=2000,
    valid_sets=[dtrain, dval],
    callbacks=[
        lgb.early_stopping(100),
        lgb.log_evaluation(200)
    ]
)

print(f"\n  最佳迭代: {lgb_model.best_iteration}")
print(f"  验证RMSE: {lgb_model.best_score['valid_1']['rmse']:.4f}")


step4_metrics = {
    'best_iteration': lgb_model.best_iteration,
    'max_rounds': 2000,
    'train_rmse': lgb_model.best_score['training']['rmse'],
    'val_rmse': lgb_model.best_score['valid_1']['rmse'],
    'n_train': len(X_train),
    'n_val': len(X_val),
    'n_features': len(feature_cols),
    'note': 'RMSE on log1p-transformed target'
}
pd.DataFrame([step4_metrics]).to_csv(
    OUTPUT_DIR / "step4_model_metrics.csv", index=False, encoding='utf-8-sig')

importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("\n  Top 15 特征重要性:")
print(importance.head(15).to_string(index=False))

del X, X_train, X_val, y_train, y_val, train_features, train_data
gc.collect()


print("\n[Step4] 6/8 全量OD流预测...")



daily_tourism = pd.read_csv(step3_path, usecols=['city_name', 'date', 'tourists_10k'])
daily_tourism['date'] = pd.to_datetime(daily_tourism['date'])
daily_arrivals = (daily_tourism.groupby(['city_name', 'date'])['tourists_10k']
                  .sum().to_dict())
del daily_tourism
gc.collect()

print("  预构建OD对与静态特征...")
n_cities = len(cities_339)


_o_grid = np.repeat(np.arange(n_cities, dtype=np.int32), n_cities)
_d_grid = np.tile(np.arange(n_cities, dtype=np.int32), n_cities)
_off_diag = _o_grid != _d_grid
origin_codes = _o_grid[_off_diag]
dest_codes = _d_grid[_off_diag]
del _o_grid, _d_grid, _off_diag
n_od_pairs = len(origin_codes)
print(f"  OD对数: {n_od_pairs}")


dist_vec = dist_matrix[origin_codes, dest_codes].astype(np.float64)


def _city_year_vec(src, city_col, val_col, year):

    if val_col not in src.columns:
        return np.full(n_cities, np.nan)
    sub = src.loc[src['year'] == year, [city_col, val_col]]
    m = dict(zip(sub[city_col].to_numpy(), sub[val_col].to_numpy()))
    return np.array([m.get(c, np.nan) for c in cities_339], dtype=np.float64)


def _year_climate_grid(year, year_dates):

    n_days = len(year_dates)
    date_pos = {d: i for i, d in enumerate(year_dates)}
    grids = {c: np.full((n_days, n_cities), np.nan)
             for c in ('tmean_c', 'precip_mm', 'precip_heavy')}
    cy = climate_all[climate_all['flow_date'].dt.year == year]
    di = cy['flow_date'].map(date_pos)
    ci = cy['city_name'].map(city_to_idx)
    ok = (di.notna() & ci.notna()).to_numpy()
    di = di.to_numpy()[ok].astype(int)
    ci = ci.to_numpy()[ok].astype(int)
    for c in grids:
        grids[c][di, ci] = cy[c].to_numpy()[ok]
    return grids



_pred_col_names = (
    ['distance_km', 'dest_tmean', 'dest_precip', 'orig_tmean', 'orig_precip_heavy',
     'temp_diff', 'month', 'day_of_week',
     'dest_annual_tourists', 'orig_huji_pop', 'orig_gdp_per_capita']
    + [f'dest_{c}' for c in ['scenic_n', 'museum_n', 'lodging_n', 'resort_n',
                             'gdp', 'tertiary_industry']]
    + [c for c in cal_features_s4 if c in calendar_df.columns])
_missing_pred_cols = [c for c in feature_cols if c not in _pred_col_names]
if _missing_pred_cols:
    raise RuntimeError(f"预测特征构建缺少训练特征: {_missing_pred_cols}")


def predict_year(year):

    print(f"\n  === {year}年 ===")
    year_dates = pd.date_range(f'{year}-01-01', f'{year}-12-31')
    date_pos = {d: i for i, d in enumerate(year_dates)}

    clim_grid = _year_climate_grid(year, year_dates)
    tmean_grid = clim_grid['tmean_c']
    precip_grid = clim_grid['precip_mm']
    pheavy_grid = clim_grid['precip_heavy']

    
    dest_static = {}
    for col in ['scenic_n', 'museum_n', 'lodging_n', 'resort_n']:
        dest_static[f'dest_{col}'] = _city_year_vec(poi_df, 'city', col, year)
    for col in ['gdp', 'tertiary_industry']:
        dest_static[f'dest_{col}'] = _city_year_vec(city_panel, 'city_name', col, year)
    dest_static['dest_annual_tourists'] = _city_year_vec(
        annual_df, 'city_name', 'tourists_final_10k', year)
    orig_static = {
        'orig_huji_pop': _city_year_vec(city_panel, 'city_name', 'huji_pop', year),
        'orig_gdp_per_capita': _city_year_vec(city_panel, 'city_name',
                                              'gdp_per_capita', year),
    }

    year_cal = calendar_df[calendar_df['flow_date'].dt.year == year].set_index('flow_date')
    year_cal = year_cal[~year_cal.index.duplicated(keep='first')]
    cal_cols_use = [c for c in cal_features_s4 if c in year_cal.columns]

    out_file = OUTPUT_DIR / f"od_tourism_flow_{year}.csv"
    if out_file.exists():
        out_file.unlink()

    n_rows_total = 0
    total_flow_year = 0.0
    first_write = True

    for month in range(1, 13):
        month_dates = year_dates[year_dates.month == month]
        month_flows = []

        for date in month_dates:
            di = date_pos[date]

            col_data = {'distance_km': dist_vec}
            for k, v in dest_static.items():
                col_data[k] = v[dest_codes]
            for k, v in orig_static.items():
                col_data[k] = v[origin_codes]

            t_day = tmean_grid[di]
            col_data['dest_tmean'] = t_day[dest_codes]
            col_data['orig_tmean'] = t_day[origin_codes]
            col_data['dest_precip'] = precip_grid[di][dest_codes]
            col_data['orig_precip_heavy'] = pheavy_grid[di][origin_codes]
            
            col_data['temp_diff'] = col_data['orig_tmean'] - col_data['dest_tmean']

            if date in year_cal.index:
                cal_row = year_cal.loc[date]
                for c in cal_cols_use:
                    col_data[c] = np.full(n_od_pairs, float(cal_row[c]))
            else:
                for c in cal_cols_use:
                    col_data[c] = np.zeros(n_od_pairs)

            col_data['month'] = np.full(n_od_pairs, date.month, dtype=np.float64)
            col_data['day_of_week'] = np.full(n_od_pairs, date.dayofweek,
                                              dtype=np.float64)

            X_pred = np.empty((n_od_pairs, len(feature_cols)), dtype=np.float32)
            for j, c in enumerate(feature_cols):
                X_pred[:, j] = col_data[c]
            del col_data

            y_pred = np.maximum(np.expm1(lgb_model.predict(X_pred)), 0)
            del X_pred

            
            targets = np.array([daily_arrivals.get((c, date), 0.0)
                                for c in cities_339], dtype=np.float64)
            targets = np.where(np.isfinite(targets) & (targets > 0), targets, 0.0)
            col_sums = np.bincount(dest_codes, weights=y_pred, minlength=n_cities)
            scale = np.zeros(n_cities)
            ok = (col_sums > 0) & (targets > 0)
            scale[ok] = targets[ok] / col_sums[ok]
            
            
            month_flows.append((y_pred * scale[dest_codes]).astype(np.float32))
            del y_pred

        n_days_m = len(month_dates)
        flow_concat = np.concatenate(month_flows)
        date_strs = np.array([d.strftime('%Y-%m-%d') for d in month_dates])
        month_result = pd.DataFrame({
            'origin': pd.Categorical.from_codes(np.tile(origin_codes, n_days_m),
                                                categories=cities_339),
            'dest': pd.Categorical.from_codes(np.tile(dest_codes, n_days_m),
                                              categories=cities_339),
            'date': np.repeat(date_strs, n_od_pairs),
            'flow_10k': flow_concat
        })
        
        month_result.to_csv(out_file, mode='w' if first_write else 'a',
                            header=first_write, index=False,
                            encoding='utf-8-sig' if first_write else 'utf-8')

        month_total = float(flow_concat.sum(dtype=np.float64))
        n_rows_total += len(flow_concat)
        total_flow_year += month_total
        print(f"    {month:02d}月: 总流量 {month_total:.1f} 万人次")
        first_write = False
        del month_flows, flow_concat, month_result
        gc.collect()

    return n_rows_total, total_flow_year



print("\n[Step4] 7/8 逐年预测并保存...")

existing_years = [y for y in YEARS if (OUTPUT_DIR / f"od_tourism_flow_{y}.csv").exists()]

if SKIP_PREDICT_IF_EXISTS and len(existing_years) == len(YEARS):
    
    
    print(f"  跳过预测: {len(existing_years)} 个年度文件已存在 ({YEARS[0]}-{YEARS[-1]})")
    print("  如需强制重新生成, 把顶部 SKIP_PREDICT_IF_EXISTS 改为 False")
else:
    if existing_years:
        print(f"  已存在 {len(existing_years)}/{len(YEARS)} 个年度文件, 将全部重新生成")
    for year in YEARS:
        n_rows, total_flow = predict_year(year)
        out_file = OUTPUT_DIR / f"od_tourism_flow_{year}.csv"
        print(f"  {year}: {n_rows:,} 条, 总流量 {total_flow:.0f} 万人次 -> {out_file.name}")
        gc.collect()


print("\n[Step4] 8/8 保存模型信息...")


importance.to_csv(OUTPUT_DIR / "od_flow_feature_importance.csv", index=False, encoding='utf-8-sig')






model_path = OUTPUT_DIR / "od_flow_lgbm_model.txt"
model_path.write_text(
    lgb_model.model_to_string(num_iteration=lgb_model.best_iteration),
    encoding='utf-8'
)

feature_doc = """
特征选择体系说明 (Push-Pull-Resistance Model)
==============================================

1. 空间阻力 (Resistance) - 1个特征
   - distance_km: 球面距离

2. 目的地拉力 (Pull) - 9个特征
   POI: dest_scenic_n, dest_museum_n, dest_lodging_n, dest_resort_n
   经济: dest_gdp, dest_tertiary_industry
   规模: dest_annual_tourists
   气候: dest_tmean, dest_precip

3. 出发地推力 (Push) - 4个特征
   社经: orig_huji_pop, orig_gdp_per_capita
   气候: orig_tmean, orig_precip_heavy

4. 时间调节 (Temporal) - 14个特征
   month, day_of_week + 12个日历特征

5. 气候交互 (Climate Interaction) - 1个特征
   temp_diff

共计: 29个特征
"""
with open(OUTPUT_DIR / "feature_selection_doc.txt", 'w', encoding='utf-8') as f:
    f.write(feature_doc)




total_time = time.time() - t0
print("\n\n")
print("=" * 70)
print("Pipeline 完成!")
print("=" * 70)
print(f"  总耗时: {total_time/60:.1f} 分钟")
print(f"  输出目录: {OUTPUT_DIR}")
print(f"")
print(f"  Step 1 产出:")
print(f"    - annual_tourism_panel_339x10.csv (339城市×10年 年度面板)")
print(f"    - ols_model_summary.txt")
print(f"  Step 3 产出:")
print(f"    - daily_tourism_panel_339_v2.csv (339城市×3652天 日度面板)")
print(f"    - step3_feature_importance.csv (LightGBM特征重要度)")
print(f"  Step 4 产出:")
print(f"    - od_tourism_flow_{{year}}.csv (2012-2021, 逐日OD流)")
print(f"    - od_flow_lgbm_model.txt")
print(f"    - od_flow_feature_importance.csv")
print(f"    - feature_selection_doc.txt")
print("=" * 70)
