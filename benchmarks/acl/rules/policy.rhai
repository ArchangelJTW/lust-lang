fn check(role, tenants, owners, teams, archived) {
 if role == 0 { if tenants / 100 == tenants % 100 { return 1; } }
 else if role == 1 && archived == 0 {
  if owners / 100 == owners % 100 { return 1; }
  if teams / 100 != 0 && teams / 100 == teams % 100 { return 1; }
 }
 0
}
fn check_object(c) {
 if c.role == 0 { if c.tenants / 100 == c.tenants % 100 { return 1; } }
 else if c.role == 1 && c.archived == 0 {
  if c.owners / 100 == c.owners % 100 { return 1; }
  if c.teams / 100 != 0 && c.teams / 100 == c.teams % 100 { return 1; }
 }
 0
}
