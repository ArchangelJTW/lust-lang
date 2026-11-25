extern crate alloc;
extern crate lust_aot_rt;

#[inline]
pub fn dot_sum(n: i64) -> Result<i64, lust_aot_rt::AotError> {
    let mut ctx = lust_aot_rt::AotCtx::new();
    dot_sum_with_ctx(&mut ctx, n)
}

#[inline]
pub fn dot_sum_with_ctx(ctx: &mut lust_aot_rt::AotCtx, n: i64) -> Result<i64, lust_aot_rt::AotError> {
    let mut acc: i64 = 0;
    let mut i: i64 = 1;
    while i <= n {
        ctx.charge_gas(1)?;
        acc = acc.wrapping_add(i.wrapping_mul(i.wrapping_add(1)));
        i += 1;
    }
    Ok(acc)
}
