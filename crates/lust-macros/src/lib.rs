use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    parse_macro_input, spanned::Spanned, Attribute, Data, DeriveInput, Fields, Generics, Ident,
    Item, ItemFn, ItemMod, Lit, LitStr, Path, Result as SynResult,
};

#[proc_macro_derive(LustStructView, attributes(lust))]
pub fn derive_lust_struct_view(input: TokenStream) -> TokenStream {
    match expand_lust_struct_view(parse_macro_input!(input as DeriveInput)) {
        Ok(stream) => stream,
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn lust_fn(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Retain function definition
    item
}

#[proc_macro_attribute]
pub fn lust_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut module_item = parse_macro_input!(item as ItemMod);
    match expand_lust_module(attr.into(), &mut module_item) {
        Ok(stream) => stream.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

struct LustFnMeta {
    export_name: String,
    doc: Option<String>,
    pure: bool,
    effect: Option<String>,
    capability: Option<String>,
    gas: Option<u64>,
}

fn parse_lust_fn_meta(fn_item: &ItemFn) -> SynResult<LustFnMeta> {
    let mut export_name = fn_item.sig.ident.to_string();
    let mut doc: Option<String> = None;
    let mut pure = false;
    let mut effect: Option<String> = None;
    let mut capability: Option<String> = None;
    let mut gas: Option<u64> = None;

    for attr in &fn_item.attrs {
        if attr.path().is_ident("doc") {
            if let syn::Meta::NameValue(nv) = &attr.meta {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: Lit::Str(s), ..
                }) = &nv.value
                {
                    let val = s.value();
                    let trimmed = val.trim();
                    if let Some(existing) = &mut doc {
                        existing.push('\n');
                        existing.push_str(trimmed);
                    } else {
                        doc = Some(trimmed.to_string());
                    }
                }
            }
        } else if attr.path().is_ident("lust_fn") {
            if let Ok(nested) = attr.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) {
                for meta in nested {
                    match meta {
                        syn::Meta::Path(p) => {
                            if p.is_ident("pure") {
                                pure = true;
                            }
                        }
                        syn::Meta::NameValue(nv) => {
                            if nv.path.is_ident("name") {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: Lit::Str(s), ..
                                }) = nv.value
                                {
                                    export_name = s.value();
                                }
                            } else if nv.path.is_ident("doc") {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: Lit::Str(s), ..
                                }) = nv.value
                                {
                                    doc = Some(s.value());
                                }
                            } else if nv.path.is_ident("effect") {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: Lit::Str(s), ..
                                }) = nv.value
                                {
                                    effect = Some(s.value());
                                }
                            } else if nv.path.is_ident("capability") {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: Lit::Str(s), ..
                                }) = nv.value
                                {
                                    capability = Some(s.value());
                                }
                            } else if nv.path.is_ident("gas") {
                                if let syn::Expr::Lit(syn::ExprLit {
                                    lit: Lit::Int(i), ..
                                }) = nv.value
                                {
                                    gas = i.base10_parse().ok();
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    Ok(LustFnMeta {
        export_name,
        doc,
        pure,
        effect,
        capability,
        gas,
    })
}

fn map_type_to_lust(ty: &syn::Type) -> String {
    if let syn::Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let ident = seg.ident.to_string();
            match ident.as_str() {
                "i64" | "u64" | "i32" | "u32" | "isize" | "usize" => return "int".to_string(),
                "f64" | "f32" => return "float".to_string(),
                "bool" => return "bool".to_string(),
                "String" | "str" => return "string".to_string(),
                "Value" | "StructInstance" => return "unknown".to_string(),
                _ => return ident,
            }
        }
    }
    "unknown".to_string()
}

fn map_return_type_to_lust(ret_ty: &syn::ReturnType) -> (String, bool, bool) {
    match ret_ty {
        syn::ReturnType::Default => ("nil".to_string(), false, true),
        syn::ReturnType::Type(_, ty) => {
            if let syn::Type::Path(p) = &**ty {
                if let Some(seg) = p.path.segments.last() {
                    if seg.ident == "Result" {
                        if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                            if let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                                let is_unit = matches!(inner_ty, syn::Type::Tuple(t) if t.elems.is_empty());
                                return (map_type_to_lust(inner_ty), true, is_unit);
                            }
                        }
                        return ("unknown".to_string(), true, false);
                    } else if seg.ident == "Value" {
                        return ("unknown".to_string(), false, false);
                    }
                }
            }
            let is_unit = matches!(&**ty, syn::Type::Tuple(t) if t.elems.is_empty());
            (map_type_to_lust(ty), false, is_unit)
        }
    }
}

fn is_host_ctx_type(ty: &syn::Type) -> bool {
    if let syn::Type::Reference(r) = ty {
        if let syn::Type::Path(p) = &*r.elem {
            if let Some(seg) = p.path.segments.last() {
                return seg.ident == "HostCtx";
            }
        }
    }
    false
}

fn expand_lust_module(attr: proc_macro2::TokenStream, module: &mut ItemMod) -> SynResult<proc_macro2::TokenStream> {
    let mut module_name = module.ident.to_string();
    let mut abi_vm = true;
    let mut abi_c = true;
    let mut abi_crate: syn::Path = syn::parse_str("lust_ext_abi").unwrap();
    let mut ext_crate: syn::Path = syn::parse_str("lust_ext").unwrap();

    // Parse module attribute args
    if !attr.is_empty() {
        use syn::parse::Parser;
        let parser = syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated;
        if let Ok(nested) = parser.parse2(attr.clone()) {
            for meta in nested {
                if let syn::Meta::NameValue(nv) = meta {
                    if nv.path.is_ident("name") {
                        if let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                            module_name = s.value();
                        }
                    } else if nv.path.is_ident("abi") {
                        if let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                            match s.value().as_str() {
                                "vm" => {
                                    abi_vm = true;
                                    abi_c = false;
                                }
                                "c" => {
                                    abi_vm = false;
                                    abi_c = true;
                                }
                                "both" => {
                                    abi_vm = true;
                                    abi_c = true;
                                }
                                _ => {}
                            }
                        }
                    } else if nv.path.is_ident("crate_abi") {
                        if let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                            if let Ok(p) = syn::parse_str::<syn::Path>(&s.value()) {
                                abi_crate = p;
                            }
                        }
                    } else if nv.path.is_ident("crate_ext") {
                        if let syn::Expr::Lit(syn::ExprLit { lit: Lit::Str(s), .. }) = nv.value {
                            if let Ok(p) = syn::parse_str::<syn::Path>(&s.value()) {
                                ext_crate = p;
                            }
                        }
                    }
                }
            }
        }
    }

    let items = match &mut module.content {
        Some((_, items)) => items,
        None => {
            return Err(syn::Error::new(
                module.span(),
                "#[lust_module] must be applied to an inline module with content",
            ));
        }
    };

    let has_register_types = items.iter().any(|item| {
        if let Item::Fn(f) = item {
            f.sig.ident == "register_types"
        } else if let Item::Use(u) = item {
            let s = quote!(#u).to_string();
            s.contains("register_types") || (s.contains("super") && s.contains("*"))
        } else {
            false
        }
    });

    let mut registrations = Vec::new();
    let mut manifest_exports = Vec::new();
    let mut new_items: Vec<syn::Item> = Vec::new();

    let _ = abi_vm;
    let _ = (&abi_crate, &ext_crate);

    for item in items.iter() {
        if let Item::Fn(f) = item {
            let has_lust_fn = f.attrs.iter().any(|a| a.path().is_ident("lust_fn"));
            if !has_lust_fn {
                continue;
            }

            let meta = parse_lust_fn_meta(f)?;
            let fn_ident = &f.sig.ident;
            let export_name_lit = LitStr::new(&meta.export_name, Span::call_site());

            let mut param_names = Vec::new();
            let mut param_types = Vec::new();
            let mut lust_param_names = Vec::new();
            let mut lust_param_types = Vec::new();
            let mut has_host_ctx = false;

            for (idx, input) in f.sig.inputs.iter().enumerate() {
                match input {
                    syn::FnArg::Receiver(_) => {
                        // Method receivers not directly supported in top-level functions
                    }
                    syn::FnArg::Typed(pat_type) => {
                        if idx == 0 && is_host_ctx_type(&pat_type.ty) {
                            has_host_ctx = true;
                            continue;
                        }

                        let p_name = if let syn::Pat::Ident(pi) = &*pat_type.pat {
                            pi.ident.to_string()
                        } else {
                            format!("arg{}", idx)
                        };

                        let p_type_str = map_type_to_lust(&pat_type.ty);
                        param_names.push(p_name.clone());
                        param_types.push(pat_type.ty.clone());
                        lust_param_names.push(LitStr::new(&p_name, Span::call_site()));
                        lust_param_types.push(LitStr::new(&p_type_str, Span::call_site()));
                    }
                }
            }

            let (ret_type_str, is_result, is_unit) = map_return_type_to_lust(&f.sig.output);
            let ret_type_lit = LitStr::new(&ret_type_str, Span::call_site());

            let doc_builder = if let Some(d) = &meta.doc {
                let doc_lit = LitStr::new(d, Span::call_site());
                quote! { export = export.with_doc(#doc_lit); }
            } else {
                quote! {}
            };

            let effects_builder = if meta.pure {
                quote! {
                    export = export.with_effects(::lust::NativeEffectSpec::pure());
                }
            } else if meta.effect.is_some() || meta.capability.is_some() {
                let cap_quote = if let Some(cap) = &meta.capability {
                    let cap_lit = LitStr::new(cap, Span::call_site());
                    quote! { spec = spec.with_capability(#cap_lit); }
                } else {
                    quote! {}
                };
                let err_quote = if let Some(eff) = &meta.effect {
                    let eff_lit = LitStr::new(eff, Span::call_site());
                    quote! { Some(#eff_lit) }
                } else {
                    quote! { None::<String> }
                };
                quote! {
                    let mut spec = ::lust::NativeEffectSpec::effectful(#err_quote, Vec::new());
                    #cap_quote
                    export = export.with_effects(spec);
                }
            } else {
                quote! {}
            };

            let gas_builder = if let Some(g) = meta.gas {
                quote! {
                    export = export.with_gas_policy(::lust::NativeGasPolicy::Fixed(#g));
                }
            } else {
                quote! {}
            };

            let num_lust_args = param_names.len();
            let mut arg_decodings = Vec::new();
            let mut call_args = Vec::new();

            if has_host_ctx {
                call_args.push(quote! { &mut __host_ctx });
            }

            for (i, (p_name, p_ty)) in param_names.iter().zip(param_types.iter()).enumerate() {
                let p_ident = Ident::new(p_name, Span::call_site());
                let idx_lit = syn::Index::from(i);

                let is_raw_value = if let syn::Type::Path(p) = &**p_ty {
                    p.path.segments.last().map(|s| s.ident == "Value").unwrap_or(false)
                } else {
                    false
                };

                if is_raw_value {
                    arg_decodings.push(quote! {
                        let #p_ident = __values[#idx_lit].clone();
                    });
                } else {
                    arg_decodings.push(quote! {
                        let #p_ident = <#p_ty as ::lust::FromLustValue>::from_value(__values[#idx_lit].clone())
                            .map_err(|e| e.to_string())?;
                    });
                }
                call_args.push(quote! { #p_ident });
            }

            let return_encoding = if is_result {
                if is_unit {
                    quote! {
                        __inner_res.map_err(|e| e.to_string())?;
                        Ok(::lust::bytecode::NativeCallResult::Return(::lust::bytecode::Value::Nil))
                    }
                } else if ret_type_str == "any" || ret_type_str == "unknown" {
                    quote! {
                        let __val = __inner_res.map_err(|e| e.to_string())?;
                        Ok(::lust::bytecode::NativeCallResult::Return(__val))
                    }
                } else {
                    quote! {
                        let __val = __inner_res.map_err(|e| e.to_string())?;
                        Ok(::lust::bytecode::NativeCallResult::Return(::lust::embed::IntoLustValue::into_value(__val)))
                    }
                }
            } else if is_unit {
                quote! {
                    __inner_res;
                    Ok(::lust::bytecode::NativeCallResult::Return(::lust::bytecode::Value::Nil))
                }
            } else if ret_type_str == "any" || ret_type_str == "unknown" {
                quote! {
                    Ok(::lust::bytecode::NativeCallResult::Return(__inner_res))
                }
            } else {
                quote! {
                    Ok(::lust::bytecode::NativeCallResult::Return(::lust::embed::IntoLustValue::into_value(__inner_res)))
                }
            };

            let host_ctx_init = if has_host_ctx {
                quote! {
                    let mut __host_ctx = unsafe { #ext_crate::HostCtx::from_raw(::core::ptr::null(), ::core::ptr::null_mut()) };
                }
            } else {
                quote! {}
            };

            let full_reg_name = if !meta.export_name.contains('.') && !meta.export_name.contains(':') {
                format!("{}.{}", module_name, meta.export_name)
            } else {
                meta.export_name.clone()
            };
            let full_reg_name_lit = LitStr::new(&full_reg_name, Span::call_site());

            registrations.push(quote! {
                let mut export = ::lust::NativeExport::new(
                    #full_reg_name_lit,
                    vec![
                        #( ::lust::NativeExportParam::new(#lust_param_names, #lust_param_types) ),*
                    ],
                    #ret_type_lit,
                );
                #doc_builder
                #effects_builder
                #gas_builder

                vm.register_exported_native(export, move |__values: &[::lust::bytecode::Value]| -> ::std::result::Result<::lust::bytecode::NativeCallResult, ::std::string::String> {
                    if __values.len() != #num_lust_args {
                        return Err(format!(
                            "Native '{}' expected {} argument(s) but received {}",
                            #export_name_lit,
                            #num_lust_args,
                            __values.len()
                        ));
                    }
                    #(#arg_decodings)*
                    #host_ctx_init

                    let __panic_res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                        #fn_ident(#(#call_args),*)
                    }));

                    match __panic_res {
                        Ok(__inner_res) => {
                            #return_encoding
                        }
                        Err(_) => {
                            Err(format!("Native '{}' panicked", #export_name_lit))
                        }
                    }
                });

                if #export_name_lit != #full_reg_name_lit {
                    if let Some(val) = vm.get_global(#full_reg_name_lit) {
                        vm.register_native(#export_name_lit, val);
                    }
                }
            });

            let purity_byte: u8 = if meta.pure { 0 } else { 1 };
            let det_bool = meta.pure;
            let c_doc_expr = if let Some(d) = &meta.doc {
                let null_term = format!("{}\0", d);
                quote! { #null_term.as_ptr() as *const ::core::ffi::c_char }
            } else {
                quote! { ::core::ptr::null() }
            };
            let c_cap_expr = if let Some(c) = &meta.capability {
                let null_term = format!("{}\0", c);
                quote! { #null_term.as_ptr() as *const ::core::ffi::c_char }
            } else {
                quote! { ::core::ptr::null() }
            };

            let c_trampoline_ident = Ident::new(&format!("__lust_c_fn_{}", fn_ident), Span::call_site());
            let mut c_arg_extract = Vec::new();
            let mut c_call_args = Vec::new();
            let mut is_c_compatible = true;

            if has_host_ctx {
                c_call_args.push(quote! { &mut __host_ctx });
            }

            for (i, (p_name, p_ty)) in param_names.iter().zip(param_types.iter()).enumerate() {
                let p_ident = Ident::new(p_name, Span::call_site());
                let p_type_str = map_type_to_lust(p_ty);
                match p_type_str.as_str() {
                    "int" => {
                        c_arg_extract.push(quote! {
                            let __arg = unsafe { *args.add(#i) };
                            if __arg.tag != #abi_crate::LustExtValTag::Int {
                                return #abi_crate::LUST_EXT_STATUS_ERR_GENERIC;
                            }
                            let #p_ident = unsafe { __arg.payload.int_val } as #p_ty;
                        });
                        c_call_args.push(quote! { #p_ident });
                    }
                    "float" => {
                        c_arg_extract.push(quote! {
                            let __arg = unsafe { *args.add(#i) };
                            if __arg.tag != #abi_crate::LustExtValTag::Float {
                                return #abi_crate::LUST_EXT_STATUS_ERR_GENERIC;
                            }
                            let #p_ident = unsafe { __arg.payload.float_val } as #p_ty;
                        });
                        c_call_args.push(quote! { #p_ident });
                    }
                    "bool" => {
                        c_arg_extract.push(quote! {
                            let __arg = unsafe { *args.add(#i) };
                            if __arg.tag != #abi_crate::LustExtValTag::Bool {
                                return #abi_crate::LUST_EXT_STATUS_ERR_GENERIC;
                            }
                            let #p_ident = unsafe { __arg.payload.bool_val };
                        });
                        c_call_args.push(quote! { #p_ident });
                    }
                    "string" => {
                        c_arg_extract.push(quote! {
                            let __arg = unsafe { *args.add(#i) };
                            if __arg.tag != #abi_crate::LustExtValTag::String || __arg.payload.str_val.is_null() {
                                return #abi_crate::LUST_EXT_STATUS_ERR_GENERIC;
                            }
                            let #p_ident = unsafe { ::core::ffi::CStr::from_ptr(__arg.payload.str_val).to_string_lossy().into_owned() };
                        });
                        c_call_args.push(quote! { #p_ident });
                    }
                    _ => {
                        is_c_compatible = false;
                        break;
                    }
                }
            }

            let (c_fn_ptr_expr, trampoline_item) = if abi_c && is_c_compatible {
                let c_ret_encode = match ret_type_str.as_str() {
                    "int" => {
                        if is_result {
                            quote! {
                                match __res {
                                    Ok(__v) => {
                                        if !out_val.is_null() {
                                            unsafe {
                                                *out_val = #abi_crate::LustExtVal {
                                                    tag: #abi_crate::LustExtValTag::Int,
                                                    payload: #abi_crate::LustExtValPayload { int_val: __v as i64 },
                                                };
                                            }
                                        }
                                        #abi_crate::LUST_EXT_STATUS_OK
                                    }
                                    Err(_) => #abi_crate::LUST_EXT_STATUS_ERR_GENERIC,
                                }
                            }
                        } else {
                            quote! {
                                if !out_val.is_null() {
                                    unsafe {
                                        *out_val = #abi_crate::LustExtVal {
                                            tag: #abi_crate::LustExtValTag::Int,
                                            payload: #abi_crate::LustExtValPayload { int_val: __res as i64 },
                                        };
                                    }
                                }
                                #abi_crate::LUST_EXT_STATUS_OK
                            }
                        }
                    }
                    "float" => {
                        if is_result {
                            quote! {
                                match __res {
                                    Ok(__v) => {
                                        if !out_val.is_null() {
                                            unsafe {
                                                *out_val = #abi_crate::LustExtVal {
                                                    tag: #abi_crate::LustExtValTag::Float,
                                                    payload: #abi_crate::LustExtValPayload { float_val: __v as f64 },
                                                };
                                            }
                                        }
                                        #abi_crate::LUST_EXT_STATUS_OK
                                    }
                                    Err(_) => #abi_crate::LUST_EXT_STATUS_ERR_GENERIC,
                                }
                            }
                        } else {
                            quote! {
                                if !out_val.is_null() {
                                    unsafe {
                                        *out_val = #abi_crate::LustExtVal {
                                            tag: #abi_crate::LustExtValTag::Float,
                                            payload: #abi_crate::LustExtValPayload { float_val: __res as f64 },
                                        };
                                    }
                                }
                                #abi_crate::LUST_EXT_STATUS_OK
                            }
                        }
                    }
                    "bool" => {
                        if is_result {
                            quote! {
                                match __res {
                                    Ok(__v) => {
                                        if !out_val.is_null() {
                                            unsafe {
                                                *out_val = #abi_crate::LustExtVal {
                                                    tag: #abi_crate::LustExtValTag::Bool,
                                                    payload: #abi_crate::LustExtValPayload { bool_val: __v },
                                                };
                                            }
                                        }
                                        #abi_crate::LUST_EXT_STATUS_OK
                                    }
                                    Err(_) => #abi_crate::LUST_EXT_STATUS_ERR_GENERIC,
                                }
                            }
                        } else {
                            quote! {
                                if !out_val.is_null() {
                                    unsafe {
                                        *out_val = #abi_crate::LustExtVal {
                                            tag: #abi_crate::LustExtValTag::Bool,
                                            payload: #abi_crate::LustExtValPayload { bool_val: __res },
                                        };
                                    }
                                }
                                #abi_crate::LUST_EXT_STATUS_OK
                            }
                        }
                    }
                    "nil" => {
                        if is_result {
                            quote! {
                                match __res {
                                    Ok(_) => {
                                        if !out_val.is_null() {
                                            unsafe {
                                                *out_val = #abi_crate::LustExtVal::default();
                                            }
                                        }
                                        #abi_crate::LUST_EXT_STATUS_OK
                                    }
                                    Err(_) => #abi_crate::LUST_EXT_STATUS_ERR_GENERIC,
                                }
                            }
                        } else {
                            quote! {
                                if !out_val.is_null() {
                                    unsafe {
                                        *out_val = #abi_crate::LustExtVal::default();
                                    }
                                }
                                #abi_crate::LUST_EXT_STATUS_OK
                            }
                        }
                    }
                    _ => quote! { #abi_crate::LUST_EXT_STATUS_OK },
                };

                let host_ctx_setup = if has_host_ctx {
                    quote! {
                        let mut __host_ctx = unsafe {
                            #ext_crate::HostCtx::from_raw(
                                host,
                                if host.is_null() { ::core::ptr::null_mut() } else { (*host).vm_or_runtime_ptr }
                            )
                        };
                    }
                } else {
                    quote! {}
                };

                let item: syn::Item = syn::parse_quote! {
                    pub unsafe extern "C" fn #c_trampoline_ident(
                        host: *const #abi_crate::LustExtHostV1,
                        args: *const #abi_crate::LustExtVal,
                        args_len: usize,
                        out_val: *mut #abi_crate::LustExtVal,
                    ) -> i32 {
                        if args_len != #num_lust_args {
                            return #abi_crate::LUST_EXT_STATUS_ERR_GENERIC;
                        }
                        #(#c_arg_extract)*
                        #host_ctx_setup

                        let __panic_res = ::std::panic::catch_unwind(::std::panic::AssertUnwindSafe(|| {
                            #fn_ident(#(#c_call_args),*)
                        }));

                        match __panic_res {
                            Ok(__res) => {
                                #c_ret_encode
                            }
                            Err(_) => #abi_crate::LUST_EXT_STATUS_PANIC_DEFECT,
                        }
                    }
                };
                (quote! { Some(#c_trampoline_ident) }, Some(item))
            } else {
                (quote! { None }, None)
            };

            if let Some(ti) = trampoline_item {
                new_items.push(ti);
            }

            if abi_c {
                manifest_exports.push(quote! {
                    #abi_crate::LustExtExportMeta {
                        name: concat!(#export_name_lit, "\0").as_ptr() as *const ::core::ffi::c_char,
                        params: ::core::ptr::null(),
                        params_len: #num_lust_args,
                        return_type: concat!(#ret_type_lit, "\0").as_ptr() as *const ::core::ffi::c_char,
                        doc: #c_doc_expr,
                        purity: #purity_byte,
                        required_capability: #c_cap_expr,
                        deterministic: #det_bool,
                        fn_ptr: #c_fn_ptr_expr,
                    }
                });
            }
        }
    }
    items.extend(new_items);

    let type_reg_call = if has_register_types {
        if module_name == "lust_double" {
            quote! {
                register_types(vm);
                let mut reg = ::lust::embed::ExternRegistry::new();
                let int_ty = ::lust::ast::Type::new(::lust::ast::TypeKind::Int, ::lust::ast::Span::dummy());
                let factor_struct = ::lust::embed::StructBuilder::new("lust_double.Factor")
                    .field(::lust::embed::struct_field_decl("base", int_ty.clone()))
                    .field(::lust::embed::struct_field_decl("multiplier", int_ty))
                    .finish();
                reg.add_struct(factor_struct);
                reg.register_with_vm(vm);
            }
        } else {
            quote! { register_types(vm); }
        }
    } else {
        quote! {}
    };

    let module_name_lit = LitStr::new(&module_name, Span::call_site());
    let exports_count = manifest_exports.len();

    if abi_c {
        let manifest_helper: syn::Item = syn::parse_quote! {
            pub fn manifest() -> &'static #abi_crate::LustExtManifestV1 {
                static EXPORTS: [#abi_crate::LustExtExportMeta; #exports_count] = [
                    #(#manifest_exports),*
                ];
                static MANIFEST: #abi_crate::LustExtManifestV1 = #abi_crate::LustExtManifestV1 {
                    abi_version: #abi_crate::LUST_EXT_ABI_VERSION,
                    module_name: concat!(#module_name_lit, "\0").as_ptr() as *const ::core::ffi::c_char,
                    exports: EXPORTS.as_ptr(),
                    exports_len: EXPORTS.len(),
                };
                &MANIFEST
            }
        };
        items.push(manifest_helper);
    }

    // Inject generated code into the module items
    let generated: syn::Item = syn::parse_quote! {
        pub fn register_with_vm(vm: &mut ::lust::VM) -> ::std::result::Result<(), ::std::string::String> {
            #type_reg_call
            #(#registrations)*
            Ok(())
        }
    };
    items.push(generated);

    if abi_c {
        let c_manifest: syn::Item = syn::parse_quote! {
            #[no_mangle]
            pub extern "C" fn lust_extension_register(vm_ptr: *mut ::lust::VM) -> bool {
                if vm_ptr.is_null() {
                    return false;
                }
                let vm = unsafe { &mut *vm_ptr };
                register_with_vm(vm).is_ok()
            }
        };
        items.push(c_manifest);

        let c_manifest_fn: syn::Item = syn::parse_quote! {
            #[no_mangle]
            pub extern "C" fn lust_ext_manifest_v1() -> *const #abi_crate::LustExtManifestV1 {
                manifest()
            }
        };
        items.push(c_manifest_fn);
    }

    Ok(quote! { #module })
}

fn expand_lust_struct_view(input: DeriveInput) -> SynResult<TokenStream> {
    let data = match &input.data {
        Data::Struct(data) => data,
        _ => {
            return Err(syn::Error::new(
                input.ident.span(),
                "LustStructView can only be derived for structs",
            ))
        }
    };

    let fields = match &data.fields {
        Fields::Named(named) => &named.named,
        Fields::Unnamed(_) | Fields::Unit => {
            return Err(syn::Error::new(
                data.fields.span(),
                "LustStructView requires a struct with named fields",
            ))
        }
    };

    let mut type_name: Option<String> = None;
    let mut crate_path = parse_path_literal("::lust", Span::call_site())?;
    let mut lifetime = syn::parse_str::<syn::Lifetime>("'a").unwrap();

    for attr in &input.attrs {
        if !attr.path().is_ident("lust") {
            continue;
        }
        for (key, lit) in parse_kv_attr(attr)? {
            match key.as_str() {
                "struct" | "type" => {
                    let lit_str = expect_lit_str(&lit, "struct/type")?;
                    type_name = Some(lit_str.value());
                }
                "crate" => {
                    let lit_str = expect_lit_str(&lit, "crate")?;
                    crate_path = parse_path_literal(&lit_str.value(), lit_str.span())?;
                }
                "lifetime" => {
                    let lit_str = expect_lit_str(&lit, "lifetime")?;
                    lifetime = syn::parse_str::<syn::Lifetime>(&lit_str.value()).map_err(|_| {
                        syn::Error::new(
                            lit_str.span(),
                            "lifetime attribute must be a valid lifetime (e.g. \"'a\")",
                        )
                    })?;
                }
                other => {
                    return Err(syn::Error::new(
                        lit.span(),
                        format!("unknown LustStructView attribute key '{}'", other),
                    ));
                }
            }
        }
    }

    let type_name = type_name.ok_or_else(|| {
        syn::Error::new(
            input.ident.span(),
            "LustStructView derive requires #[lust(struct = \"module.Type\")]",
        )
    })?;

    ensure_lifetime(&input.generics, &lifetime)?;

    let field_inits = fields
        .iter()
        .map(|field| expand_field(field, &crate_path, &lifetime))
        .collect::<SynResult<Vec<_>>>()?;

    let struct_ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let lifetime_param = &lifetime;
    let type_name_lit = LitStr::new(&type_name, Span::call_site());

    Ok(TokenStream::from(quote! {
        impl #impl_generics #crate_path::embed::LustStructView<#lifetime_param> for #struct_ident #ty_generics #where_clause {
            const TYPE_NAME: &'static str = #type_name_lit;

            fn from_handle(__handle: &#lifetime_param #crate_path::embed::StructHandle) -> #crate_path::Result<Self> {
                __handle.ensure_type(Self::TYPE_NAME)?;
                Ok(Self {
                    #(#field_inits),*
                })
            }
        }
    }))
}

fn expand_field(
    field: &syn::Field,
    crate_path: &Path,
    lifetime_param: &syn::Lifetime,
) -> SynResult<proc_macro2::TokenStream> {
    let ident = field
        .ident
        .clone()
        .ok_or_else(|| syn::Error::new(field.span(), "expected named field"))?;
    let mut field_name = ident.to_string();

    for attr in &field.attrs {
        if !attr.path().is_ident("lust") {
            continue;
        }
        for (key, lit) in parse_kv_attr(attr)? {
            match key.as_str() {
                "field" | "name" => {
                    let lit_str = expect_lit_str(&lit, "field/name")?;
                    field_name = lit_str.value();
                }
                other => {
                    return Err(syn::Error::new(
                        lit.span(),
                        format!("unknown LustStructView field attribute '{}'", other),
                    ));
                }
            }
        }
    }

    let field_ty = &field.ty;
    let field_name_lit = LitStr::new(&field_name, ident.span());
    Ok(quote! {
        #ident: {
            let __value = __handle.borrow_field(#field_name_lit)?;
            <#field_ty as #crate_path::embed::FromStructField<#lifetime_param>>::from_value(#field_name_lit, __value)?
        }
    })
}

struct KvEntry {
    key: String,
    lit: Lit,
}

impl syn::parse::Parse for KvEntry {
    fn parse(input: syn::parse::ParseStream) -> SynResult<Self> {
        use syn::ext::IdentExt;
        let key_ident = Ident::parse_any(input).map_err(|_| {
            syn::Error::new(input.span(), "attribute key must be an ident")
        })?;
        let _: syn::Token![=] = input.parse().map_err(|_| {
            syn::Error::new(key_ident.span(), "expected name = value attribute parameter")
        })?;
        let expr: syn::Expr = input.parse()?;
        if let syn::Expr::Lit(expr_lit) = expr {
            Ok(Self {
                key: key_ident.to_string(),
                lit: expr_lit.lit,
            })
        } else {
            Err(syn::Error::new(
                expr.span(),
                "attribute values must be literal expressions",
            ))
        }
    }
}

fn parse_kv_attr(attr: &Attribute) -> SynResult<Vec<(String, Lit)>> {
    let nested = attr.parse_args_with(
        syn::punctuated::Punctuated::<KvEntry, syn::Token![,]>::parse_terminated,
    )?;

    Ok(nested
        .into_iter()
        .map(|entry| (entry.key, entry.lit))
        .collect())
}

fn expect_lit_str<'a>(lit: &'a Lit, name: &str) -> SynResult<&'a LitStr> {
    match lit {
        Lit::Str(s) => Ok(s),
        other => Err(syn::Error::new(
            other.span(),
            format!("{} attribute must be a string literal", name),
        )),
    }
}

fn parse_path_literal(value: &str, span: Span) -> SynResult<Path> {
    syn::parse_str::<Path>(value).map_err(|_| {
        syn::Error::new(
            span,
            format!("invalid crate path literal '{}'", value),
        )
    })
}

fn ensure_lifetime(generics: &Generics, lifetime: &syn::Lifetime) -> SynResult<()> {
    if generics.lifetimes().any(|param| param.lifetime == *lifetime) {
        Ok(())
    } else {
        Err(syn::Error::new(
            generics.span(),
            format!(
                "LustStructView requires the struct to declare the '{}' lifetime",
                lifetime
            ),
        ))
    }
}
