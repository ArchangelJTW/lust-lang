use std::env;
fn main(){let Some(p)=env::args().nth(1)else{std::process::exit(2)};match p.as_str(){
"fields"=>{let(mut x,mut y)=(0_i64,0_i64);for i in 1..=10_000_000{x=i;y+=i}let _=x;println!("{y}")},
"array"=>{let a:Vec<i64>=(0..1_000_000).collect();let mut s=0_i64;for _ in 0..10{for &v in &a{s+=v}}println!("{s}")},
"sieve"=>{let n=10_000_000usize;let mut f=vec![true;n+1];let mut c=0;for i in 2..=n{if f[i]{c+=1;let mut j=i*i;while j<=n{f[j]=false;j+=i}}}println!("{c}")},
"calls"=>{fn step(a:i64,b:i64)->i64{a+b*2}let mut a=0;for i in 0..10_000_000{a=step(a,i)}println!("{a}")},
"methods"=>{struct C{i:i64}impl C{fn bump(&mut self,d:i64)->i64{self.i+=d;self.i}}let mut c=C{i:0};let mut last=0;for i in 0..10_000_000{last=c.bump(i)}println!("{last}")},
"strings"=>{let mut s=String::new();for i in 0..100_000{s.push(char::from(b'0'+(i%10)as u8))}println!("{}",s.len())},
"fib"=>{fn fib(n:i64)->i64{if n<2{n}else{fib(n-1)+fib(n-2)}}println!("{}",fib(30))},
"nested"=>{let mut t=0_i64;for i in 0..3000_i64{for j in 0..3000_i64{t+=(i*j)%7}}println!("{t}")},
"floatmath"=>{let(mut x,mut a)=(0.0_f64,0.0_f64);for _ in 0..10_000_000{x+=0.001;a+=x*x-a*0.5}println!("{a}")},
"tree"=>{struct N{v:i64,l:Option<Box<N>>,r:Option<Box<N>>}fn build(d:i64,v:i64)->N{if d==0{N{v,l:None,r:None}}else{N{v,l:Some(Box::new(build(d-1,v*2))),r:Some(Box::new(build(d-1,v*2+1)))}}}fn sum(n:&N)->i64{n.v+n.l.as_deref().map_or(0,sum)+n.r.as_deref().map_or(0,sum)}let tr=build(14,1);let t=(0..20).map(|_|sum(&tr)).sum::<i64>();println!("{}",t+1_000_000)},_=>std::process::exit(2)}}
