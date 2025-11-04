use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{
    parse_macro_input, spanned::Spanned, Attribute, Data, DeriveInput, Fields, Generics, Ident,
    Lit, LitStr, Path, Result as SynResult,
};

#[proc_macro_derive(LustStructView, attributes(lust))]
pub fn derive_lust_struct_view(input: TokenStream) -> TokenStream {
    match expand_lust_struct_view(parse_macro_input!(input as DeriveInput)) {
        Ok(stream) => stream,
        Err(err) => err.to_compile_error().into(),
    }
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

fn parse_kv_attr(attr: &Attribute) -> SynResult<Vec<(String, Lit)>> {
    attr.parse_args_with(|input: syn::parse::ParseStream| {
        let mut pairs = Vec::new();
        while !input.is_empty() {
            let key = if input.peek(syn::Token![type]) {
                input.parse::<syn::Token![type]>()?;
                "type".to_string()
            } else if input.peek(syn::Token![struct]) {
                input.parse::<syn::Token![struct]>()?;
                "struct".to_string()
            } else if input.peek(syn::Token![crate]) {
                input.parse::<syn::Token![crate]>()?;
                "crate".to_string()
            } else {
                let ident: Ident = input.parse()?;
                ident.to_string()
            };
            input.parse::<syn::Token![=]>()?;
            let value: Lit = input.parse()?;
            pairs.push((key, value));
            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
        }
        Ok(pairs)
    })
}

fn expect_lit_str<'a>(lit: &'a Lit, context: &str) -> SynResult<&'a LitStr> {
    match lit {
        Lit::Str(s) => Ok(s),
        _ => Err(syn::Error::new(
            lit.span(),
            format!("{context} attribute expects a string literal"),
        )),
    }
}

fn parse_path_literal(src: &str, span: Span) -> SynResult<Path> {
    syn::parse_str(src)
        .map_err(|_| syn::Error::new(span, format!("unable to parse '{}' as a path literal", src)))
}

fn ensure_lifetime(generics: &Generics, lifetime: &syn::Lifetime) -> SynResult<()> {
    let expected = lifetime.ident.to_string();
    let found = generics
        .lifetimes()
        .any(|lt| lt.lifetime.ident == lifetime.ident);
    if found {
        Ok(())
    } else {
        Err(syn::Error::new(
            generics.span(),
            format!(
                "LustStructView derive expects the struct to declare lifetime {}",
                expected
            ),
        ))
    }
}
