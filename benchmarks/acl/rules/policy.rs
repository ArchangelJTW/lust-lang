#[derive(Clone, Copy)]
pub struct Context { pub role:i64, pub tenants:i64, pub owners:i64, pub teams:i64, pub archived:i64 }
#[inline]
pub fn check(role:i64, tenants:i64, owners:i64, teams:i64, archived:i64)->i64 {
 if (role==0 && tenants/100==tenants%100) || (role==1 && archived==0 && (owners/100==owners%100 || (teams/100!=0 && teams/100==teams%100))) {1} else {0}
}
#[inline]
pub fn check_object(c:Context)->i64 {
 if (c.role==0 && c.tenants/100==c.tenants%100) || (c.role==1 && c.archived==0 && (c.owners/100==c.owners%100 || (c.teams/100!=0 && c.teams/100==c.teams%100))) {1} else {0}
}
