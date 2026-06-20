# -*- coding: utf-8 -*-
"""발행 타이밍 버스트 분석: 매체·기자별 peak60 + 무작위 발행시점 영모형 검정."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
try: import koreanize_matplotlib
except: pass

df=pd.read_csv("bodies_20240913_고려아연.csv")
df["t"]=pd.to_datetime(df["published_at"],errors="coerce")
df=df.dropna(subset=["t"]).copy()
sec_all=df["t"].values.astype("datetime64[s]").astype("int64")
TMIN,TMAX=sec_all.min(),sec_all.max()
SPAN=TMAX-TMIN
def peak60(sec):
    sec=np.sort(sec);
    if len(sec)==0: return 0
    best=1;j=0
    for i in range(len(sec)):
        while sec[i]-sec[j]>3600: j+=1
        best=max(best,i-j+1)
    return int(best)
rng=np.random.default_rng(7)
def null_p(n,obs,nsim=3000):
    if n<2 or obs<2: return 1.0
    cnt=0
    for _ in range(nsim):
        s=rng.integers(TMIN,TMAX+1,size=n)
        if peak60(s)>=obs: cnt+=1
    return (cnt+1)/(nsim+1)

def table(grpcol, minn):
    rows=[]
    for k,g in df.groupby(grpcol):
        sec=g["t"].values.astype("datetime64[s]").astype("int64")
        n=len(sec); pk=peak60(sec)
        if n<minn: continue
        rows.append(dict(name=str(k),total=n,peak60=pk,conc=round(pk/n,2),p=null_p(n,pk)))
    return pd.DataFrame(rows).sort_values(["peak60","total"],ascending=False)

# 매체
mp=table("press",5)
print("[매체별] total/peak60/집중도/영모형p")
print(mp.head(16).to_string(index=False))

# 기자 (개인명만; 데스크·통합 제외)
df["rep"]=df["reporter"].apply(lambda x: str(x).strip())
PRESSES=set(df["press"].astype(str))
bad=["닷컴","뉴스","팀","부서","편집","온라인","속보","데스크","종합","미상","nan",
     "칼럼","오피니언","영상","기고","객원","이벤트","신청","사진","그래픽","스크","소문",
     "늦었","운영","관리","경제","일보","미디어","투데이","비즈"]
def is_person(r):
    nm=str(r).replace("기자","").strip()
    if len(nm)!=3: return False                 # 한국 기자명 대부분 3글자
    if nm in PRESSES: return False
    return not any(b in str(r) for b in bad)
df_r=df[df["rep"].apply(is_person)]
rp=[]
for k,g in df_r.groupby("rep"):
    sec=g["t"].values.astype("datetime64[s]").astype("int64")
    n=len(sec); pk=peak60(sec)
    if n<4: continue
    rp.append(dict(name=k,press=g["press"].iloc[0],total=n,peak60=pk,conc=round(pk/n,2),p=null_p(n,pk)))
rp=pd.DataFrame(rp).sort_values(["peak60","total"],ascending=False)
print("\n[기자별 rapid-fire 후보 (n>=4)] total/peak60/집중도/p")
print(rp.to_string(index=False))

# 분류
mp["type"]=np.where(mp.conc>=0.55,"버스트형",np.where(mp.total>=11,"물량형","혼합"))
res=dict(span_h=round(SPAN/3600,1),
         media=mp.to_dict("records"), reporters=rp.to_dict("records"))
json.dump(res,open("_burst_results.json","w",encoding="utf-8"),ensure_ascii=False,indent=1)

# 그림: total vs peak60 산점도(매체) + 영모형 유의 표시
fig,ax=plt.subplots(1,2,figsize=(15,6))
sig=mp.p<0.05
ax[0].scatter(mp.total[~sig],mp.peak60[~sig],s=60,c="#9DB4CE",label="비유의")
ax[0].scatter(mp.total[sig],mp.peak60[sig],s=80,c="#E0552B",label="영모형 p<0.05")
for _,r in mp.iterrows():
    if r.peak60>=4 or r.total>=12:
        ax[0].annotate(r["name"],(r.total,r.peak60),fontsize=8,xytext=(3,3),textcoords="offset points")
for c,lab in [(0.7,"집중도 0.7"),(0.4,"0.4"),(0.25,"0.25")]:
    xs=np.array([4,17]); ax[0].plot(xs,xs*c,ls=":",c="#bbb",lw=1)
ax[0].set_xlabel("총 기사 수 (total)"); ax[0].set_ylabel("1시간 최대 (peak60)")
ax[0].set_title("매체별 발행 패턴 — 물량형(우하) vs 버스트형(좌상)"); ax[0].legend(fontsize=9)
# 기자 막대
top=rp.head(10).iloc[::-1]
y=range(len(top))
ax[1].barh(y,top.total,color="#C9D6E5",label="총 기사")
ax[1].barh(y,top.peak60,color="#E0552B",label="1시간 최대(peak60)")
ax[1].set_yticks(y); ax[1].set_yticklabels([f"{r['name']}·{r['press']}" for _,r in top.iterrows()],fontsize=9)
for i,(_,r) in enumerate(top.iterrows()):
    ax[1].text(r.total+0.1,i,f"p={r.p:.3f}",va="center",fontsize=8,color="#444")
ax[1].set_xlabel("기사 수"); ax[1].set_title("기자별 rapid-fire (60분 내 집중)"); ax[1].legend(fontsize=9,loc="lower right")
plt.tight_layout(); plt.savefig("report/assets/burst_kz.png",dpi=120)
print("\n→ report/assets/burst_kz.png, _burst_results.json 저장")
print("검증 김준형:",rp[rp.name.str.contains('김준형')][['total','peak60','p']].to_dict('records'))
