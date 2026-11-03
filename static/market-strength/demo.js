// Seeded synthetic series, deliberately unrelated to actual market prices.
export function demoData(){
 let seed=18273;const rand=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
 const groups=['Healthcare','Telecommunication','Capital Goods','Textiles','Information Technology','Financial Services','Consumer Goods','Metals & Mining'];
 const dates=[];let d=new Date('2024-09-02T00:00:00Z');while(dates.length<560){if(d.getUTCDay()!==0&&d.getUTCDay()!==6)dates.push(d.toISOString().slice(0,10));d.setUTCDate(d.getUTCDate()+1);}
 const universe=Array.from({length:40},(_,i)=>{let p=70+i*32;return {symbol:`DEMO${String(i+1).padStart(2,'0')}`,name:`${['Aster','Cedar','Meridian','Pioneer','Summit'][i%5]} ${['Labs','Networks','Industries','Textiles','Systems','Finance','Consumer','Metals'][i%8]}`,sector:groups[i%8],bars:dates.map((date,j)=>{const open=p,drift=.0001+(i%8)*.00023+.0014*Math.sin(j/85+i),noise=(rand()-.49)*.045;p=Math.max(5,p*(1+drift+noise));const high=Math.max(open,p)*(1+rand()*.014),low=Math.min(open,p)*(1-rand()*.014);return {date,open,high,low,close:p,volume:Math.round(1000000+rand()*9000000)};})};});
 let p=20000;const benchmark=dates.map((date,i)=>{const open=p;p*=1+.00035+Math.sin(i/60)*.0004+(rand()-.5)*.012;return {date,open,close:p,high:Math.max(p,open)*1.002,low:Math.min(p,open)*.998,volume:1};});return {universe,benchmark};
}
