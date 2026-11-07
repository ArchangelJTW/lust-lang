local function check(role, tenants, owners, teams, archived)
 if role == 0 then if math.floor(tenants/100) == tenants%100 then return 1 end
 elseif role == 1 and archived == 0 then
  if math.floor(owners/100) == owners%100 then return 1 end
  if math.floor(teams/100) ~= 0 and math.floor(teams/100) == teams%100 then return 1 end
 end
 return 0
end
local function check_object(c)
 if c.role == 0 then if math.floor(c.tenants/100) == c.tenants%100 then return 1 end
 elseif c.role == 1 and c.archived == 0 then
  if math.floor(c.owners/100) == c.owners%100 then return 1 end
  if math.floor(c.teams/100) ~= 0 and math.floor(c.teams/100) == c.teams%100 then return 1 end
 end
 return 0
end
return {check=check,check_object=check_object}
